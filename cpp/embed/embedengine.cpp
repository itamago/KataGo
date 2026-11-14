#include "../embed/embedinternal.h"

#include "../core/config_parser.h"
#include "../core/global.h"
#include "../core/logger.h"
#include "../core/rand.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/nninputs.h"
#include "../program/play.h"
#include "../program/playutils.h"
#include "../program/setup.h"
#include "../search/asyncbot.h"
#include "../search/search.h"

#include "onnx.pb.h"
#include <onnxruntime_cxx_api.h>
#include <dml_provider_factory.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <mutex>
#include <streambuf>

using namespace std;
using namespace katago_embed;
using namespace katago_embed::internal;

//--------------------------------------------------------------
// Log lines of KataGo's Logger, forwarded to the application

namespace katago_embed {
namespace internal {

class LogLineBuffer : public std::streambuf {
 public:
  std::function<void(const string&)> onLog;

 protected:
  int overflow(int c) override {
    if(c == EOF)
      return c;
    std::lock_guard<std::mutex> lock(mutexLine);
    if(c == '\n') {
      if(onLog && !line.empty())
        onLog(line);
      line.clear();
    }
    else if(c != '\r') {
      line.push_back((char)c);
    }
    return c;
  }

  std::streamsize xsputn(const char* s, std::streamsize n) override {
    for(std::streamsize i = 0; i < n; i++)
      overflow((unsigned char)s[i]);
    return n;
  }

 private:
  std::mutex mutexLine;
  string line;
};

}  // namespace internal
}  // namespace katago_embed

//--------------------------------------------------------------

struct Engine::Impl {
  EngineConfig config;
  map<string, string> keysBase;
  ConfigParser* cfg = NULL;
  LogLineBuffer logBuffer;
  ostream logStream;
  Logger* logger = NULL;
  Rand* seedRand = NULL;
  NNEvaluator* nnEval = NULL;
  NNEvaluator* humanEval = NULL;
  AsyncBot* bot = NULL;
  int searchId = 0;

  // The position given to setPosition, constant while a search runs
  GameSetup setup;
  vector<Move> moves;
  int size = 19;

  mutable std::mutex mutexInfo;
  EngineInfo info;

  Impl() : logStream(&logBuffer) {}

  void release() {
    if(bot != NULL)
      bot->stopAndWait();
    delete bot;
    bot = NULL;
    delete humanEval;
    humanEval = NULL;
    delete nnEval;
    nnEval = NULL;
    delete logger;
    logger = NULL;
    delete cfg;
    cfg = NULL;
    delete seedRand;
    seedRand = NULL;
  }

  // Search parameters : the base keys, plus the keys of a play style
  SearchParams loadParams(const map<string, string>& keysStyle) const {
    map<string, string> keys = keysBase;
    for(const auto& pair: keysStyle)
      keys[pair.first] = pair.second;
    ConfigParser cfgParams(keys);
    return Setup::loadSingleParams(cfgParams, Setup::SETUP_FOR_GTP, humanEval != NULL);
  }

  map<string, string> keysForStyle(const PlayStyle& style) const {
    map<string, string> keys;
    keys["maxVisits"] = Global::int64ToString(std::max<int64_t>(1, style.maxVisits));
    if(style.maxTime > 0.0)
      keys["maxTime"] = Global::doubleToString(style.maxTime);

    // Human-like play, the values of KataGo's gtp_human5k_example.cfg
    if(!style.profileHuman.empty() && humanEval != NULL) {
      keys["humanSLProfile"] = style.profileHuman;
      keys["humanSLChosenMoveProp"] = "1.0";
      keys["humanSLChosenMoveIgnorePass"] = "true";
      keys["humanSLChosenMovePiklLambda"] = "100000000";
      keys["humanSLRootExploreProbWeightless"] = "0.0";
      keys["humanSLRootExploreProbWeightful"] = "0.0";
      keys["humanSLPlaExploreProbWeightless"] = "0.0";
      keys["humanSLPlaExploreProbWeightful"] = "0.0";
      keys["humanSLOppExploreProbWeightless"] = "0.0";
      keys["humanSLOppExploreProbWeightful"] = "0.0";
      keys["humanSLCpuctExploration"] = "0.50";
      keys["humanSLCpuctPermanent"] = "0.2";
      keys["chosenMoveTemperatureEarly"] = "0.85";
      keys["chosenMoveTemperature"] = "0.70";
      keys["chosenMoveTemperatureHalflife"] = "80";
      keys["chosenMoveTemperatureOnlyBelowProb"] = "0.01";
      keys["chosenMoveSubtract"] = "0";
      keys["chosenMovePrune"] = "0";
      keys["ignorePreRootHistory"] = "false";
      keys["rootNumSymmetriesToSample"] = "2";
      keys["useLcbForSelection"] = "false";
      keys["winLossUtilityFactor"] = "1.0";
      keys["staticScoreUtilityFactor"] = "0.30";
      keys["dynamicScoreUtilityFactor"] = "0.00";
      keys["useUncertainty"] = "false";
      keys["subtreeValueBiasFactor"] = "0.0";
      keys["useNoisePruning"] = "false";
    }
    return keys;
  }

  // Rebuild a position in KataGo's types
  bool buildPosition(const GameSetup& setupPos, const vector<Move>& movesPos, Board& board, Player& pla, BoardHistory& hist, string& errorMsg) const {
    BoardHistoryModes modes = Search::resolveHistoryModes(bot->getParams(), nnEval);
    if(!buildInitial(setupPos, modes, board, pla, hist, errorMsg))
      return false;
    for(const Move& move: movesPos) {
      if(!applyMove(board, hist, pla, move, errorMsg))
        return false;
    }
    return true;
  }

  // What a search knows, from Black's point of view. Safe while the search runs.
  Analysis makeAnalysis(const Search* search, int countCandidates) const {
    Analysis analysis;
    const Board& board = search->getRootBoard();
    const int boardSize = board.x_size;
    analysis.moveNumber = (int)moves.size();
    analysis.toPlay = search->getRootPla() == P_WHITE ? Stone::White : Stone::Black;

    // Root
    ReportedSearchValues values;
    if(search->getRootValues(values)) {
      analysis.winrateBlack = 1.0 - 0.5 * (1.0 + values.winLossValue);
      analysis.scoreLeadBlack = -values.lead;
      analysis.visits = values.visits;
    }

    // Candidates
    vector<AnalysisData> buf;
    search->getAnalysisData(buf, std::max(1, countCandidates), false, 20, false);
    std::sort(buf.begin(), buf.end(), [](const AnalysisData& a, const AnalysisData& b) { return a.order < b.order; });
    for(const AnalysisData& data: buf) {
      if((int)analysis.candidates.size() >= countCandidates)
        break;
      Candidate candidate;
      candidate.move = fromLoc(data.move, boardSize);
      candidate.winrateBlack = 1.0 - 0.5 * (1.0 + data.winLossValue);
      candidate.scoreLeadBlack = -data.lead;
      candidate.scoreStdev = data.scoreStdev;
      candidate.prior = data.policyPrior;
      candidate.visits = data.numVisits;
      candidate.order = data.order;
      for(Loc loc: data.pv)
        candidate.pv.push_back(fromLoc(loc, boardSize));
      analysis.candidates.push_back(candidate);
    }

    // Ownership
    vector<double> ownership = search->getAverageTreeOwnership(P_BLACK, NULL, 0);
    if((int)ownership.size() == boardSize * boardSize) {
      analysis.ownershipBlack.resize(ownership.size());
      for(size_t i = 0; i < ownership.size(); i++)
        analysis.ownershipBlack[i] = (float)ownership[i];
    }

    // Policy
    float policyProbs[NNPos::MAX_NN_POLICY_SIZE];
    if(search->getPolicy(policyProbs)) {
      analysis.policy.assign(boardSize * boardSize + 1, -1.0f);
      for(int y = 0; y < boardSize; y++) {
        for(int x = 0; x < boardSize; x++)
          analysis.policy[y * boardSize + x] = policyProbs[NNPos::xyToPos(x, y, search->nnXLen)];
      }
      analysis.policy[boardSize * boardSize] = policyProbs[NNPos::locToPos(Board::PASS_LOC, boardSize, search->nnXLen, search->nnYLen)];
    }
    return analysis;
  }

  void noteSearchSpeed(const Analysis& analysis, double seconds) {
    if(seconds <= 0.05 || analysis.visits <= 0)
      return;
    std::lock_guard<std::mutex> lock(mutexInfo);
    info.visitsPerSecond = (double)analysis.visits / seconds;
  }
};

//--------------------------------------------------------------

static double secondsSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

Engine::Engine()
  : impl(new Impl())
{}

Engine::~Engine() {
  finalize();
  delete impl;
}

bool Engine::probeDirectML(int indexAdapter, string& errorMsg) {
  try {
    // One Relu on 4 floats
    onnx::ModelProto model;
    model.set_ir_version(9);
    onnx::OperatorSetIdProto* opset = model.add_opset_import();
    opset->set_domain("");
    opset->set_version(20);
    onnx::GraphProto* graph = model.mutable_graph();
    graph->set_name("probe");
    onnx::NodeProto* node = graph->add_node();
    node->set_op_type("Relu");
    node->add_input("X");
    node->add_output("Y");
    for(int i = 0; i < 2; i++) {
      onnx::ValueInfoProto* info = i == 0 ? graph->add_input() : graph->add_output();
      info->set_name(i == 0 ? "X" : "Y");
      onnx::TypeProto_Tensor* tensor = info->mutable_type()->mutable_tensor_type();
      tensor->set_elem_type(onnx::TensorProto_DataType_FLOAT);
      tensor->mutable_shape()->add_dim()->set_dim_value(1);
      tensor->mutable_shape()->add_dim()->set_dim_value(4);
    }
    string bytes;
    model.SerializeToString(&bytes);

    // The DirectML provider, as the ONNX backend appends it
    const OrtApi* ortApi = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    const OrtDmlApi* dmlApi = NULL;
    OrtStatus* status = ortApi->GetExecutionProviderApi("DML", ORT_API_VERSION, (const void**)&dmlApi);
    if(status != NULL || dmlApi == NULL) {
      errorMsg = string("DirectML provider missing from ONNX Runtime : ") + (status != NULL ? ortApi->GetErrorMessage(status) : "no API");
      if(status != NULL)
        ortApi->ReleaseStatus(status);
      return false;
    }
    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "KataGoEmbedProbe");
    Ort::SessionOptions options;
    options.DisableMemPattern();
    options.SetExecutionMode(ORT_SEQUENTIAL);
    Ort::ThrowOnError(dmlApi->SessionOptionsAppendExecutionProvider_DML(options, indexAdapter));
    Ort::Session session(env, bytes.data(), bytes.size(), options);

    // One run
    float input[4] = {-1.0f, 0.0f, 1.0f, 2.0f};
    float output[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int64_t shape[2] = {1, 4};
    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value tensorIn = Ort::Value::CreateTensor<float>(memory, input, 4, shape, 2);
    Ort::Value tensorOut = Ort::Value::CreateTensor<float>(memory, output, 4, shape, 2);
    const char* namesIn[1] = {"X"};
    const char* namesOut[1] = {"Y"};
    session.Run(Ort::RunOptions{nullptr}, namesIn, &tensorIn, 1, namesOut, &tensorOut, 1);
    if(output[0] != 0.0f || output[3] != 2.0f) {
      errorMsg = "DirectML returned a wrong result on the probe graph";
      return false;
    }
    return true;
  }
  catch(const std::exception& e) {
    errorMsg = string("DirectML unavailable : ") + e.what();
  }
  catch(...) {
    errorMsg = "DirectML unavailable";
  }
  return false;
}

bool Engine::initialize(const EngineConfig& config, string& errorMsg) {
  finalize();
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  impl->config = config;
  {
    std::lock_guard<std::mutex> lock(impl->mutexInfo);
    impl->info = EngineInfo();
  }

  // GPU usable ? Probed first : a provider failing inside a server thread of KataGo terminates the process
  bool useGPU = config.useGPU;
  string errorGPU;
  if(useGPU && !probeDirectML(config.indexAdapter, errorGPU))
    useGPU = false;

  for(int attempt = 0; attempt < 2; attempt++) {
    try {
      // Configuration
      map<string, string>& keys = impl->keysBase;
      keys.clear();
      keys["onnxProvider"] = useGPU ? "directml" : "cpu";
      if(useGPU)
        keys["onnxDeviceToUse"] = Global::intToString(config.indexAdapter);
      keys["numSearchThreads"] = Global::intToString(std::max(1, config.countThreads));
      keys["maxVisits"] = "400";
      keys["ponderingEnabled"] = "false";
      keys["logToStdout"] = "false";
      keys["logToStderr"] = "false";
      keys["logSearchInfo"] = "false";
      keys["logConfigContents"] = "false";
      keys["nnCacheSizePowerOfTwo"] = "20";
      keys["nnMutexPoolSizePowerOfTwo"] = "16";
      if(!config.modelFileHuman.empty())
        keys["humanSLProfile"] = "preaz_5k";
      for(const auto& pair: config.keysExtra)
        keys[pair.first] = pair.second;
      impl->cfg = new ConfigParser(keys);

      // Logger to the application
      impl->logger = new Logger(impl->cfg, false, false, true, false);
      impl->logger->addOStream(impl->logStream);
      impl->seedRand = new Rand();

      // Nets
      const int countThreads = std::max(1, config.countThreads);
      impl->nnEval = Setup::initializeNNEvaluator(
        config.modelFile, config.modelFile, "", *impl->cfg, *impl->logger, *impl->seedRand, countThreads,
        Board::MAX_LEN, Board::MAX_LEN, Setup::MaxBatchSizeRequest::fromConcurrency(), false, false, Setup::SETUP_FOR_GTP
      );
      if(!config.modelFileHuman.empty()) {
        impl->humanEval = Setup::initializeNNEvaluator(
          config.modelFileHuman, config.modelFileHuman, "", *impl->cfg, *impl->logger, *impl->seedRand, countThreads,
          Board::MAX_LEN, Board::MAX_LEN, Setup::MaxBatchSizeRequest::fromConcurrency(), false, false, Setup::SETUP_FOR_GTP
        );
      }

      // Bot
      SearchParams params = Setup::loadSingleParams(*impl->cfg, Setup::SETUP_FOR_GTP, impl->humanEval != NULL);
      impl->bot = new AsyncBot(params, impl->nnEval, impl->humanEval, impl->logger, Global::uint64ToString(impl->seedRand->nextUInt64()));
      impl->bot->setAlwaysIncludeOwnerMap(true);

      // Empty position
      string errorPos;
      GameSetup setupDefault;
      if(!setPosition(setupDefault, vector<Move>(), errorPos))
        throw StringError(errorPos);

      std::lock_guard<std::mutex> lock(impl->mutexInfo);
      impl->info.provider = useGPU ? "directml" : "cpu";
      impl->info.errorGPU = errorGPU;
      impl->info.hasHumanModel = impl->humanEval != NULL;
      impl->info.secondsLoading = secondsSince(start);
      return true;
    }
    catch(const std::exception& e) {
      errorMsg = e.what();
    }
    catch(...) {
      errorMsg = "Unknown error";
    }
    impl->release();

    // The GPU failed : once more on the CPU
    if(!useGPU)
      break;
    errorGPU = errorMsg;
    useGPU = false;
  }
  return false;
}

void Engine::finalize() {
  try {
    impl->release();
  }
  catch(...) {
  }
}

bool Engine::isInitialized() const {
  return impl->bot != NULL;
}

bool Engine::setPosition(const GameSetup& setup, const vector<Move>& moves, string& errorMsg) {
  if(impl->bot == NULL) {
    errorMsg = "Engine not initialized";
    return false;
  }
  try {
    impl->bot->stopAndWait();
    Board board;
    Player pla;
    BoardHistory hist;
    if(!impl->buildPosition(setup, moves, board, pla, hist, errorMsg))
      return false;
    impl->bot->setPosition(pla, board, hist);
    impl->setup = setup;
    impl->moves = moves;
    impl->size = setup.size;
    return true;
  }
  catch(const std::exception& e) {
    errorMsg = e.what();
  }
  return false;
}

bool Engine::genMoveAsync(const PlayStyle& style, std::function<void(const Move&, bool, const Analysis&)> onMove, string& errorMsg) {
  if(impl->bot == NULL) {
    errorMsg = "Engine not initialized";
    return false;
  }
  try {
    impl->bot->stopAndWait();
    impl->bot->setParams(impl->loadParams(impl->keysForStyle(style)));
    Player pla = impl->bot->getRootPla();
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    Impl* state = impl;
    const bool allowResign = style.allowResign;
    impl->bot->genMoveAsync(pla, ++impl->searchId, TimeControls(), 1.0,
      [state, onMove, start, pla, allowResign](Loc moveLoc, int searchId, Search* search) {
        (void)searchId;
        Analysis analysis = state->makeAnalysis(search, 8);
        analysis.isFinal = true;
        state->noteSearchSpeed(analysis, secondsSince(start));

        // Resign when hopeless : the bot's winrate under 3 %, more than 15 points behind, after a quarter of the board
        const double winrateBot = pla == P_BLACK ? analysis.winrateBlack : 1.0 - analysis.winrateBlack;
        const double leadBot = pla == P_BLACK ? analysis.scoreLeadBlack : -analysis.scoreLeadBlack;
        const bool lateEnough = analysis.moveNumber > (state->size * state->size) / 4;
        const bool isResign = allowResign && lateEnough && winrateBot < 0.03 && leadBot < -15.0;

        Move move = moveLoc == Board::NULL_LOC ? Move::pass() : fromLoc(moveLoc, state->size);
        onMove(move, isResign, analysis);
      }
    );
    return true;
  }
  catch(const std::exception& e) {
    errorMsg = e.what();
  }
  return false;
}

bool Engine::analyzeAsync(int64_t maxVisits, double periodSeconds, int countCandidates, std::function<void(const Analysis&)> onAnalysis, string& errorMsg) {
  if(impl->bot == NULL) {
    errorMsg = "Engine not initialized";
    return false;
  }
  try {
    impl->bot->stopAndWait();
    map<string, string> keys;
    keys["maxVisits"] = Global::int64ToString(std::max<int64_t>(1, maxVisits));
    keys["wideRootNoise"] = "0.04";
    impl->bot->setParams(impl->loadParams(keys));
    Player pla = impl->bot->getRootPla();
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    Impl* state = impl;
    const int count = std::max(1, countCandidates);

    // A search with periodic reports, then a final one
    impl->bot->genMoveAsyncAnalyze(pla, ++impl->searchId, TimeControls(), 1.0,
      [state, onAnalysis, start, count](Loc moveLoc, int searchId, Search* search) {
        (void)moveLoc;
        (void)searchId;
        Analysis analysis = state->makeAnalysis(search, count);
        analysis.isFinal = true;
        state->noteSearchSpeed(analysis, secondsSince(start));
        onAnalysis(analysis);
      },
      periodSeconds, periodSeconds,
      [state, onAnalysis, start, count](const Search* search) {
        Analysis analysis = state->makeAnalysis(search, count);
        state->noteSearchSpeed(analysis, secondsSince(start));
        onAnalysis(analysis);
      }
    );
    return true;
  }
  catch(const std::exception& e) {
    errorMsg = e.what();
  }
  return false;
}

bool Engine::evaluateRaw(Analysis& out, string& errorMsg) {
  out = Analysis();
  if(impl->bot == NULL) {
    errorMsg = "Engine not initialized";
    return false;
  }
  try {
    impl->bot->stopAndWait();
    Board board = impl->bot->getRootBoard();
    BoardHistory hist = impl->bot->getRootHist();
    Player pla = impl->bot->getRootPla();
    MiscNNInputParams params;
    params.symmetry = 0;
    NNResultBuf buf;
    impl->nnEval->evaluate(board, hist, pla, &impl->bot->getParams().humanSLProfile, params, buf, true, true);
    const NNOutput* output = buf.result.get();
    if(output == NULL) {
      errorMsg = "No output from the net";
      return false;
    }
    const int boardSize = board.x_size;
    out.moveNumber = (int)impl->moves.size();
    out.toPlay = pla == P_WHITE ? Stone::White : Stone::Black;
    const double sum = output->whiteWinProb + output->whiteLossProb;
    out.winrateBlack = sum > 0.0 ? output->whiteLossProb / sum : 0.5;
    out.scoreLeadBlack = -output->whiteLead;
    out.policy.assign(boardSize * boardSize + 1, -1.0f);
    for(int y = 0; y < boardSize; y++) {
      for(int x = 0; x < boardSize; x++)
        out.policy[y * boardSize + x] = output->policyProbs[NNPos::xyToPos(x, y, output->nnXLen)];
    }
    out.policy[boardSize * boardSize] = output->policyProbs[NNPos::locToPos(Board::PASS_LOC, boardSize, output->nnXLen, output->nnYLen)];
    if(output->whiteOwnerMap != NULL) {
      out.ownershipBlack.resize(boardSize * boardSize);
      for(int y = 0; y < boardSize; y++) {
        for(int x = 0; x < boardSize; x++)
          out.ownershipBlack[y * boardSize + x] = -output->whiteOwnerMap[NNPos::xyToPos(x, y, output->nnXLen)];
      }
    }
    out.isFinal = true;
    return true;
  }
  catch(const std::exception& e) {
    errorMsg = e.what();
  }
  return false;
}

bool Engine::scoreFinal(ScoreResult& out, int64_t visits, string& errorMsg) {
  out = ScoreResult();
  if(impl->bot == NULL) {
    errorMsg = "Engine not initialized";
    return false;
  }
  try {
    impl->bot->stopAndWait();

    // The position without its final passes : a search needs a game still running
    vector<Move> moves = impl->moves;
    while(!moves.empty() && moves.back().isPass)
      moves.pop_back();
    Board board;
    Player pla;
    BoardHistory hist;
    if(!impl->buildPosition(impl->setup, moves, board, pla, hist, errorMsg))
      return false;

    // Dead stones and lead, as KataGo's GTP final_status_list and final_score
    SearchParams paramsOld = impl->bot->getParams();
    Search* search = impl->bot->getSearchStopAndWait();
    const int64_t countVisits = std::max<int64_t>(50, visits);
    vector<double> ownership;
    vector<bool> isAlive = PlayUtils::computeAnticipatedStatusesWithOwnership(search, board, hist, pla, countVisits, ownership);
    double leadWhite = PlayUtils::computeLead(search, NULL, board, hist, pla, countVisits, OtherGameProperties());
    if(hist.rules.gameResultWillBeInteger())
      leadWhite = std::round(leadWhite);
    else
      leadWhite = std::round(leadWhite + 0.5) - 0.5;

    // Restore the search on the position as it was
    impl->bot->setParams(paramsOld);
    {
      Board boardBack;
      Player plaBack;
      BoardHistory histBack;
      string errorBack;
      if(impl->buildPosition(impl->setup, impl->moves, boardBack, plaBack, histBack, errorBack))
        impl->bot->setPosition(plaBack, boardBack, histBack);
    }

    // Result
    const int boardSize = board.x_size;
    out.scoreBlack = (float)-leadWhite;
    out.winner = leadWhite < 0 ? Stone::Black : leadWhite > 0 ? Stone::White : Stone::Empty;
    out.owner.assign(boardSize * boardSize, Stone::Empty);
    out.isDead.assign(boardSize * boardSize, 0);
    for(int y = 0; y < boardSize; y++) {
      for(int x = 0; x < boardSize; x++) {
        Loc loc = Location::getLoc(x, y, boardSize);
        const int index = y * boardSize + x;
        const int pos = NNPos::locToPos(loc, boardSize, search->nnXLen, search->nnYLen);
        const double ownBlack = pos < (int)ownership.size() ? -ownership[pos] : 0.0;
        const Color color = board.colors[loc];
        Stone owner = ownBlack > 0.3 ? Stone::Black : ownBlack < -0.3 ? Stone::White : Stone::Empty;
        if(color != C_EMPTY && isAlive[loc])
          owner = toStone(color);
        if(color != C_EMPTY && !isAlive[loc]) {
          out.isDead[index] = 1;
          owner = color == C_BLACK ? Stone::White : Stone::Black;
        }
        out.owner[index] = owner;
        const bool countsAsTerritory = color == C_EMPTY || out.isDead[index] != 0;
        if(countsAsTerritory && owner == Stone::Black)
          out.territoryBlack++;
        else if(countsAsTerritory && owner == Stone::White)
          out.territoryWhite++;
      }
    }
    out.fromNeuralNet = true;
    return true;
  }
  catch(const std::exception& e) {
    errorMsg = e.what();
  }
  return false;
}

void Engine::stop() {
  try {
    if(impl->bot != NULL)
      impl->bot->stopAndWait();
  }
  catch(...) {
  }
}

void Engine::setLogCallback(std::function<void(const string&)> onLog) {
  impl->logBuffer.onLog = onLog;
}

EngineInfo Engine::getInfo() const {
  std::lock_guard<std::mutex> lock(impl->mutexInfo);
  return impl->info;
}
