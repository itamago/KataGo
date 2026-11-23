// Test of the katago_embed facade.
//  katago_embed_test                         : the rules only
//  KATAGO_TEST_MODEL=<main .bin.gz>          : plus the engine on the CPU, then on DirectML
//  KATAGO_TEST_MODEL_HUMAN=<human .bin.gz>   : plus a human-like move

#include "katagoembed.h"

#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

using namespace katago_embed;

static int countFailures = 0;

#define CHECK(cond) do { if(!(cond)) { std::printf("FAILED line %d : %s\n", __LINE__, #cond); countFailures++; } } while(0)

static Stone at(const Game& game, int x, int y) {
  BoardState state;
  game.getState(state);
  return state.stones[y * state.size + x];
}

static void testRules() {
  std::string error;

  // Capture in the corner
  {
    Game game;
    GameSetup setup;
    CHECK(game.reset(setup, error));
    CHECK(game.play(Move::point(1, 0), error));
    CHECK(game.play(Move::point(0, 0), error));
    CHECK(game.play(Move::point(0, 1), error));
    BoardState state;
    game.getState(state);
    CHECK(state.stones[0] == Stone::Empty);
    CHECK(state.capturesByBlack == 1);
    CHECK(state.nextPlayer == Stone::White);
    CHECK(state.moveNumber == 3);
    CHECK(!game.play(Move::point(1, 0), error));
    CHECK(error == "Occupied");
  }

  // Ko, undo, redo
  {
    Game game;
    GameSetup setup;
    setup.setupBlack = {Move::point(1, 0), Move::point(0, 1), Move::point(1, 2)};
    setup.setupWhite = {Move::point(1, 1), Move::point(2, 0), Move::point(3, 1), Move::point(2, 2)};
    setup.firstPlayer = Stone::Black;
    CHECK(game.reset(setup, error));
    CHECK(game.play(Move::point(2, 1), error));
    CHECK(at(game, 1, 1) == Stone::Empty);
    BoardState state;
    game.getState(state);
    CHECK(state.koForbidden == Move::point(1, 1));
    CHECK(!game.isLegal(Move::point(1, 1)));
    CHECK(!game.play(Move::point(1, 1), error));
    CHECK(error.find("Ko") == 0);
    CHECK(game.undo());
    CHECK(at(game, 1, 1) == Stone::White);
    CHECK(at(game, 2, 1) == Stone::Empty);
    CHECK(game.canRedo());
    CHECK(game.redo());
    CHECK(at(game, 2, 1) == Stone::Black);
    CHECK(!game.canRedo());
    CHECK(game.getLiberties(2, 1) == 1);
  }

  // Suicide
  {
    Game game;
    GameSetup setup;
    setup.setupWhite = {Move::point(1, 0), Move::point(0, 1), Move::point(1, 1)};
    setup.firstPlayer = Stone::Black;
    CHECK(game.reset(setup, error));
    CHECK(!game.play(Move::point(0, 0), error));
    CHECK(error == "Suicide");
  }

  // Two passes end the game ; handicap gives the move to White
  {
    Game game;
    GameSetup setup;
    setup.handicap = 4;
    CHECK(game.reset(setup, error));
    BoardState state;
    game.getState(state);
    CHECK(state.nextPlayer == Stone::White);
    int countBlack = 0;
    for(Stone stone: state.stones)
      countBlack += stone == Stone::Black ? 1 : 0;
    CHECK(countBlack == 4);
    CHECK(game.play(Move::pass(), error));
    CHECK(game.play(Move::pass(), error));
    game.getState(state);
    CHECK(state.isGameOver);
    ScoreResult score;
    CHECK(game.scoreSimple(score));
    CHECK((int)score.owner.size() == 19 * 19);
  }

  // Exact count with dead stones : two walls on 9x9, each closing two eyes, a dame column between them, a dead white stone in Black's area
  {
    const char* rulesTested[2] = { "japanese", "chinese" };
    for(int indexRules = 0; indexRules < 2; indexRules++) {
      Game game;
      GameSetup setup;
      setup.size = 9;
      setup.rules = rulesTested[indexRules];
      setup.komi = indexRules == 0 ? 6.5f : 7.5f;
      for(int y = 0; y < 9; y++) {
        setup.setupBlack.push_back(Move::point(3, y));
        setup.setupWhite.push_back(Move::point(5, y));
      }
      for(int x = 0; x < 3; x++) {
        setup.setupBlack.push_back(Move::point(x, 4));
        setup.setupWhite.push_back(Move::point(6 + x, 4));
      }
      setup.setupWhite.push_back(Move::point(1, 1));
      CHECK(game.reset(setup, error));
      CHECK(game.play(Move::pass(), error));
      CHECK(game.play(Move::pass(), error));
      std::vector<uint8_t> isDead(81, 0);
      isDead[1 * 9 + 1] = 1;
      ScoreResult score;
      CHECK(game.scoreWithDead(isDead, score));
      CHECK(score.winner == Stone::White);
      CHECK(score.countBlack.territory == 24);
      CHECK(score.countWhite.territory == 24);
      CHECK(score.owner[1 * 9 + 1] == Stone::Black);
      CHECK(score.owner[4 * 9 + 4] == Stone::Empty);
      if(indexRules == 0) {
        CHECK(score.isTerritoryScoring);
        CHECK(score.countBlack.deadStones == 1);
        CHECK(std::fabs(score.countBlack.total - 25.0f) < 0.01f);
        CHECK(std::fabs(score.countWhite.total - 30.5f) < 0.01f);
        CHECK(std::fabs(score.scoreBlack + 5.5f) < 0.01f);
      }
      else {
        CHECK(!score.isTerritoryScoring);
        CHECK(score.isSameAsKataGo);
        CHECK(score.countBlack.stones == 12);
        CHECK(score.countWhite.stones == 12);
        CHECK(std::fabs(score.countWhite.total - 43.5f) < 0.01f);
        CHECK(std::fabs(score.scoreBlack + 7.5f) < 0.01f);
      }
    }
  }

  // SGF round trip
  {
    Game game;
    GameSetup setup;
    setup.komi = 7.5f;
    CHECK(game.reset(setup, error));
    CHECK(game.play(Move::point(3, 3), error));
    CHECK(game.play(Move::point(15, 15), error));
    CHECK(game.play(Move::point(15, 3), error));
    const char* temp = std::getenv("TEMP");
    std::string path = std::string(temp != NULL ? temp : ".") + "/katago_embed_test.sgf";
    CHECK(game.saveSgf(path, "Black", "White", error));
    Game loaded;
    CHECK(loaded.loadSgf(path, error));
    CHECK(loaded.getMoves().size() == 3);
    CHECK(at(loaded, 15, 3) == Stone::Black);
    CHECK(std::fabs(loaded.getSetup().komi - 7.5f) < 0.01f);
    std::remove(path.c_str());
  }
}

struct Waiter {
  std::mutex mutex;
  std::condition_variable condition;
  bool done = false;
  Move move;
  bool isResign = false;
  Analysis analysis;
  int countReports = 0;

  void wait() {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [this]() { return done; });
  }
};

static double secondsSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

static bool testEngine(bool useGPU, const std::string& model, const std::string& modelHuman, Analysis& raw) {
  std::string error;
  Engine engine;
  engine.setLogCallback([](const std::string& line) {
    if(line.find("ERROR") != std::string::npos || line.find("WARNING") != std::string::npos)
      std::printf("  [katago] %s\n", line.c_str());
  });
  EngineConfig config;
  config.modelFile = model;
  config.modelFileHuman = modelHuman;
  config.useGPU = useGPU;
  config.countThreads = 8;
  if(!engine.initialize(config, error)) {
    std::printf("  initialize failed : %s\n", error.c_str());
    CHECK(false);
    return false;
  }
  EngineInfo info = engine.getInfo();
  std::printf("  provider %s, loaded in %.1f s%s%s\n", info.provider.c_str(), info.secondsLoading, info.errorGPU.empty() ? "" : ", GPU refused : ", info.errorGPU.c_str());
  if(useGPU && info.provider != "directml")
    std::printf("  WARNING : DirectML asked, CPU running\n");

  // Walled territories : Black's lower-left (x <= 6, y >= 12), White's upper-right (x >= 12, y <= 6), Black to move
  {
    GameSetup walls;
    for(int i = 0; i <= 7; i++) {
      walls.setupBlack.push_back(Move::point(i, 11));
      walls.setupWhite.push_back(Move::point(11 + i, 7));
    }
    for(int i = 0; i <= 6; i++) {
      walls.setupBlack.push_back(Move::point(7, 12 + i));
      walls.setupWhite.push_back(Move::point(11, i));
    }
    walls.firstPlayer = Stone::Black;
    Analysis rawWalls;
    CHECK(engine.setPosition(walls, std::vector<Move>(), error));
    CHECK(engine.evaluateRaw(rawWalls, error));
    const float insideBlack = rawWalls.ownershipBlack[16 * 19 + 2];
    const float insideWhite = rawWalls.ownershipBlack[2 * 19 + 16];
    std::printf("  walls, Black to move : inside Black's %+.2f, inside White's %+.2f\n", insideBlack, insideWhite);
    CHECK(insideBlack > 0.5f);
    CHECK(insideWhite < -0.5f);

    // The same, White to move : the same owners
    walls.firstPlayer = Stone::White;
    CHECK(engine.setPosition(walls, std::vector<Move>(), error));
    CHECK(engine.evaluateRaw(rawWalls, error));
    std::printf("  walls, White to move : inside Black's %+.2f, inside White's %+.2f\n", rawWalls.ownershipBlack[16 * 19 + 2], rawWalls.ownershipBlack[2 * 19 + 16]);
    CHECK(rawWalls.ownershipBlack[16 * 19 + 2] > 0.5f);
    CHECK(rawWalls.ownershipBlack[2 * 19 + 16] < -0.5f);
  }

  // A position
  GameSetup setup;
  std::vector<Move> moves = {Move::point(3, 3), Move::point(15, 15), Move::point(15, 3)};
  CHECK(engine.setPosition(setup, moves, error));

  // Raw net
  CHECK(engine.evaluateRaw(raw, error));
  CHECK(raw.policy.size() == 19 * 19 + 1);
  CHECK(raw.ownershipBlack.size() == 19 * 19);
  std::printf("  raw : winrate Black %.4f, lead Black %.2f\n", raw.winrateBlack, raw.scoreLeadBlack);

  // Ownership map, on demand (KATAGO_TEST_PRINT) : x Black, o White ; the 3-3 point of a lone 4-4 reads as the opponent's, KataGo's usual view
  if(std::getenv("KATAGO_TEST_PRINT") != NULL) {
    for(int y = 0; y < 19; y++) {
      std::printf("   ");
      for(int x = 0; x < 19; x++) {
        const bool isStone = (x == 3 && y == 3) || (x == 15 && y == 3) || (x == 15 && y == 15);
        const float o = raw.ownershipBlack[y * 19 + x];
        std::printf("%c", isStone ? (x == 15 && y == 15 ? 'W' : 'B') : (o > 0.15f ? 'x' : (o < -0.15f ? 'o' : '.')));
      }
      std::printf("\n");
    }
  }

  // A move at full strength
  {
    Waiter waiter;
    PlayStyle style;
    style.maxVisits = 400;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    CHECK(engine.genMoveAsync(style, [&waiter](const Move& move, bool isResign, const Analysis& analysis) {
      std::lock_guard<std::mutex> lock(waiter.mutex);
      waiter.move = move;
      waiter.isResign = isResign;
      waiter.analysis = analysis;
      waiter.done = true;
      waiter.condition.notify_all();
    }, error));
    waiter.wait();
    CHECK(waiter.move.isPoint());
    CHECK(!waiter.isResign);
    CHECK(!waiter.analysis.candidates.empty());
    std::printf("  genmove 400 visits : (%d,%d) in %.2f s, %.0f visits/s, winrate Black %.3f\n",
      waiter.move.x, waiter.move.y, secondsSince(start), engine.getInfo().visitsPerSecond, waiter.analysis.winrateBlack);
  }

  // Analysis with reports
  {
    Waiter waiter;
    CHECK(engine.analyzeAsync(1500, 0.2, 5, [&waiter](const Analysis& analysis) {
      std::lock_guard<std::mutex> lock(waiter.mutex);
      waiter.countReports++;
      if(analysis.isFinal) {
        waiter.analysis = analysis;
        waiter.done = true;
        waiter.condition.notify_all();
      }
    }, error));
    waiter.wait();
    CHECK(waiter.analysis.candidates.size() == 5);
    CHECK(waiter.analysis.ownershipBlack.size() == 19 * 19);
    CHECK(!waiter.analysis.candidates[0].pv.empty());
    std::printf("  analysis : %d reports, %lld visits, best (%d,%d) %.3f, %.0f visits/s\n",
      waiter.countReports, (long long)waiter.analysis.visits, waiter.analysis.candidates[0].move.x, waiter.analysis.candidates[0].move.y,
      waiter.analysis.candidates[0].winrateBlack, engine.getInfo().visitsPerSecond);
  }

  // Human-like move
  if(!modelHuman.empty()) {
    Waiter waiter;
    PlayStyle style;
    style.profileHuman = "rank_10k";
    style.maxVisits = 40;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    CHECK(engine.genMoveAsync(style, [&waiter](const Move& move, bool isResign, const Analysis& analysis) {
      (void)isResign;
      std::lock_guard<std::mutex> lock(waiter.mutex);
      waiter.move = move;
      waiter.analysis = analysis;
      waiter.done = true;
      waiter.condition.notify_all();
    }, error));
    waiter.wait();
    CHECK(waiter.move.isPoint());
    std::printf("  human 10k move : (%d,%d) in %.2f s\n", waiter.move.x, waiter.move.y, secondsSince(start));
  }

  // Final score
  {
    ScoreResult score;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    CHECK(engine.scoreFinal(score, 100, error));
    CHECK(score.fromNeuralNet);
    CHECK((int)score.isDead.size() == 19 * 19);
    std::printf("  score : Black %+.1f in %.2f s\n", score.scoreBlack, secondsSince(start));
  }

  engine.finalize();
  return true;
}

int main() {
  std::string error;
  if(!initializeStatic(error)) {
    std::printf("%s\n", error.c_str());
    return 2;
  }

  std::printf("Rules\n");
  testRules();

  const char* model = std::getenv("KATAGO_TEST_MODEL");
  const char* modelHuman = std::getenv("KATAGO_TEST_MODEL_HUMAN");
  if(model != NULL && model[0] != '\0') {
    std::string human = modelHuman != NULL ? modelHuman : "";
    Analysis rawCPU;
    Analysis rawGPU;
    std::printf("Engine on the CPU\n");
    bool okCPU = testEngine(false, model, human, rawCPU);
    std::string errorProbe;
    std::printf("DirectML probe : %s\n", Engine::probeDirectML(0, errorProbe) ? "ok" : errorProbe.c_str());
    std::printf("Engine on DirectML\n");
    bool okGPU = testEngine(true, model, human, rawGPU);
    if(okCPU && okGPU) {
      double maxPolicy = 0.0;
      for(size_t i = 0; i < rawCPU.policy.size(); i++)
        maxPolicy = std::fmax(maxPolicy, std::fabs(rawCPU.policy[i] - rawGPU.policy[i]));
      std::printf("CPU / DirectML : winrate %.5f / %.5f, max policy difference %.5f\n", rawCPU.winrateBlack, rawGPU.winrateBlack, maxPolicy);
      CHECK(std::fabs(rawCPU.winrateBlack - rawGPU.winrateBlack) < 0.01);
      CHECK(maxPolicy < 0.01);
    }
  }

  finalizeStatic();
  std::printf(countFailures == 0 ? "ALL PASSED\n" : "%d FAILURE(S)\n", countFailures);
  return countFailures == 0 ? 0 : 1;
}
