#include "../embed/embedinternal.h"

#include "../core/fileutils.h"
#include "../core/global.h"
#include "../dataio/sgf.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nninterface.h"
#include "../program/playutils.h"

#include <fstream>

using namespace std;

//--------------------------------------------------------------
// Global tables

bool katago_embed::initializeStatic(string& errorMsg) {
  try {
    Board::initHash();
    ScoreValue::initTables();
    NeuralNet::globalInitialize();
    return true;
  }
  catch(const std::exception& e) {
    errorMsg = string("KataGo initialization failed : ") + e.what();
  }
  catch(...) {
    errorMsg = "KataGo initialization failed";
  }
  return false;
}

void katago_embed::finalizeStatic() {
  try {
    NeuralNet::globalCleanup();
    ScoreValue::freeTables();
  }
  catch(...) {
  }
}

//--------------------------------------------------------------
// Conversions

Loc katago_embed::internal::toLoc(const Move& move, int size) {
  if(move.isPass)
    return Board::PASS_LOC;
  if(move.x < 0 || move.y < 0 || move.x >= size || move.y >= size)
    return Board::NULL_LOC;
  return Location::getLoc(move.x, move.y, size);
}

katago_embed::Move katago_embed::internal::fromLoc(Loc loc, int size) {
  if(loc == Board::PASS_LOC)
    return Move::pass();
  if(loc == Board::NULL_LOC)
    return Move();
  return Move::point(Location::getX(loc, size), Location::getY(loc, size));
}

katago_embed::Stone katago_embed::internal::toStone(Color color) {
  if(color == C_BLACK)
    return Stone::Black;
  if(color == C_WHITE)
    return Stone::White;
  return Stone::Empty;
}

Player katago_embed::internal::toPlayer(Stone stone) {
  return stone == Stone::White ? P_WHITE : P_BLACK;
}

int katago_embed::internal::countTrailingPasses(const vector<Move>& moves) {
  int count = 0;
  for(size_t i = moves.size(); i > 0; i--) {
    if(!moves[i-1].isPass)
      break;
    count++;
  }
  return count;
}

bool katago_embed::internal::buildInitial(
  const GameSetup& setup,
  const BoardHistoryModes& modes,
  Board& board,
  Player& pla,
  BoardHistory& hist,
  string& errorMsg
) {
  // Size
  if(setup.size < 2 || setup.size > Board::MAX_LEN) {
    errorMsg = "Board size " + Global::intToString(setup.size) + " is not supported (2 to " + Global::intToString(Board::MAX_LEN) + ")";
    return false;
  }

  // Rules
  Rules rules;
  if(!Rules::tryParseRulesWithoutKomi(setup.rules, rules, setup.komi)) {
    errorMsg = "Unknown rules '" + setup.rules + "'";
    return false;
  }

  // Handicap
  board = Board(setup.size, setup.size);
  if(setup.handicap != 0) {
    if(setup.handicap < 2 || setup.handicap > 9) {
      errorMsg = "Handicap must be 0 or 2 to 9";
      return false;
    }
    PlayUtils::placeFixedHandicap(board, setup.handicap);
  }

  // Setup stones
  for(int i = 0; i < 2; i++) {
    const vector<Move>& stones = i == 0 ? setup.setupBlack : setup.setupWhite;
    for(const Move& move: stones) {
      Loc loc = toLoc(move, setup.size);
      if(loc == Board::NULL_LOC || loc == Board::PASS_LOC || board.colors[loc] != C_EMPTY) {
        errorMsg = "Invalid setup stone";
        return false;
      }
      if(!board.setStone(loc, i == 0 ? C_BLACK : C_WHITE)) {
        errorMsg = "Setup stone without liberty";
        return false;
      }
    }
  }

  // First player
  if(setup.firstPlayer != Stone::Empty)
    pla = toPlayer(setup.firstPlayer);
  else if(setup.handicap > 0)
    pla = P_WHITE;
  else
    pla = P_BLACK;

  hist = BoardHistory(board, pla, rules, 0, modes);
  return true;
}

bool katago_embed::internal::applyMove(Board& board, BoardHistory& hist, Player& pla, const Move& move, string& errorMsg) {
  // Finished ?
  if(hist.isGameFinished) {
    errorMsg = "The game is over";
    return false;
  }

  // Outside ?
  Loc loc = toLoc(move, board.x_size);
  if(loc == Board::NULL_LOC) {
    errorMsg = "Outside the board";
    return false;
  }

  // Illegal ? Find why
  if(!hist.isLegal(board, loc, pla)) {
    if(loc != Board::PASS_LOC && board.colors[loc] != C_EMPTY)
      errorMsg = "Occupied";
    else if(loc == board.ko_loc)
      errorMsg = "Ko : the stone cannot be retaken at once";
    else if(!board.isLegal(loc, pla, hist.rules.multiStoneSuicideLegal))
      errorMsg = "Suicide";
    else
      errorMsg = "Superko : this move repeats an earlier position";
    return false;
  }

  hist.makeBoardMoveAssumeLegal(board, loc, pla, NULL, true);
  pla = getOpp(pla);
  return true;
}

//--------------------------------------------------------------
// Game

struct katago_embed::Game::Impl {
  GameSetup setup;
  BoardHistoryModes modes;
  Board board;
  BoardHistory hist;
  Player pla = P_BLACK;
  vector<Move> moves;
  vector<Move> movesUndone;
};

using namespace katago_embed;
using namespace katago_embed::internal;

Game::Game()
  : impl(NULL)
{}

Game::~Game() {
  delete impl;
}

Game::Game(const Game& other)
  : impl(other.impl == NULL ? NULL : new Impl(*other.impl))
{}

Game& Game::operator=(const Game& other) {
  if(this == &other)
    return *this;
  delete impl;
  impl = other.impl == NULL ? NULL : new Impl(*other.impl);
  return *this;
}

bool Game::reset(const GameSetup& setup, string& errorMsg) {
  try {
    Impl* fresh = new Impl();
    fresh->setup = setup;
    if(!buildInitial(setup, fresh->modes, fresh->board, fresh->pla, fresh->hist, errorMsg)) {
      delete fresh;
      return false;
    }
    delete impl;
    impl = fresh;
    return true;
  }
  catch(const std::exception& e) {
    errorMsg = e.what();
  }
  catch(...) {
    errorMsg = "Unknown error";
  }
  return false;
}

bool Game::play(const Move& move, string& errorMsg) {
  if(impl == NULL) {
    errorMsg = "No game";
    return false;
  }
  try {
    if(!applyMove(impl->board, impl->hist, impl->pla, move, errorMsg))
      return false;
    impl->moves.push_back(move);
    impl->movesUndone.clear();
    return true;
  }
  catch(const std::exception& e) {
    errorMsg = e.what();
  }
  return false;
}

bool Game::isLegal(const Move& move) const {
  if(impl == NULL || impl->hist.isGameFinished)
    return false;
  Loc loc = toLoc(move, impl->board.x_size);
  if(loc == Board::NULL_LOC)
    return false;
  return impl->hist.isLegal(impl->board, loc, impl->pla);
}

void Game::getLegalMask(vector<uint8_t>& mask) const {
  mask.clear();
  if(impl == NULL)
    return;
  const int size = impl->board.x_size;
  mask.assign(size * size, 0);
  if(impl->hist.isGameFinished)
    return;
  for(int y = 0; y < size; y++) {
    for(int x = 0; x < size; x++) {
      Loc loc = Location::getLoc(x, y, size);
      mask[y * size + x] = impl->hist.isLegal(impl->board, loc, impl->pla) ? 1 : 0;
    }
  }
}

bool Game::undo() {
  if(impl == NULL || impl->moves.empty())
    return false;

  // Replay every move but the last from the setup, as KataGo's GTP undo does
  vector<Move> moves = impl->moves;
  Move undone = moves.back();
  moves.pop_back();
  Board board;
  BoardHistory hist;
  Player pla;
  string errorMsg;
  if(!buildInitial(impl->setup, impl->modes, board, pla, hist, errorMsg))
    return false;
  for(const Move& move: moves) {
    if(!applyMove(board, hist, pla, move, errorMsg))
      return false;
  }
  impl->board = board;
  impl->hist = hist;
  impl->pla = pla;
  impl->moves = moves;
  impl->movesUndone.push_back(undone);
  return true;
}

bool Game::redo() {
  if(impl == NULL || impl->movesUndone.empty())
    return false;
  Move move = impl->movesUndone.back();
  string errorMsg;
  if(!applyMove(impl->board, impl->hist, impl->pla, move, errorMsg))
    return false;
  impl->moves.push_back(move);
  impl->movesUndone.pop_back();
  return true;
}

bool Game::canUndo() const {
  return impl != NULL && !impl->moves.empty();
}

bool Game::canRedo() const {
  return impl != NULL && !impl->movesUndone.empty();
}

void Game::getState(BoardState& out) const {
  out = BoardState();
  if(impl == NULL)
    return;
  const Board& board = impl->board;
  const int size = board.x_size;
  out.size = size;
  out.stones.resize(size * size);
  for(int y = 0; y < size; y++) {
    for(int x = 0; x < size; x++)
      out.stones[y * size + x] = toStone(board.colors[Location::getLoc(x, y, size)]);
  }
  out.nextPlayer = impl->pla == P_WHITE ? Stone::White : Stone::Black;
  out.moveNumber = (int)impl->moves.size();
  out.capturesByBlack = board.numWhiteCaptures;
  out.capturesByWhite = board.numBlackCaptures;
  if(!impl->moves.empty())
    out.lastMove = impl->moves.back();
  if(board.ko_loc != Board::NULL_LOC)
    out.koForbidden = fromLoc(board.ko_loc, size);
  out.consecutivePasses = countTrailingPasses(impl->moves);
  out.isGameOver = impl->hist.isGameFinished || out.consecutivePasses >= 2;
}

const GameSetup& Game::getSetup() const {
  static const GameSetup none;
  return impl == NULL ? none : impl->setup;
}

const vector<katago_embed::Move>& Game::getMoves() const {
  static const vector<katago_embed::Move> none;
  return impl == NULL ? none : impl->moves;
}

int Game::getLiberties(int x, int y) const {
  if(impl == NULL)
    return 0;
  Loc loc = toLoc(Move::point(x, y), impl->board.x_size);
  if(loc == Board::NULL_LOC || impl->board.colors[loc] == C_EMPTY)
    return 0;
  return impl->board.getNumLiberties(loc);
}

bool Game::isLadderCaptured(int x, int y) const {
  if(impl == NULL)
    return false;
  Loc loc = toLoc(Move::point(x, y), impl->board.x_size);
  if(loc == Board::NULL_LOC)
    return false;
  Color owner = impl->board.colors[loc];
  if(owner != C_BLACK && owner != C_WHITE)
    return false;
  if(impl->board.getNumLiberties(loc) != 1)
    return false;

  // The opponent to move takes it at once
  if(impl->pla != owner)
    return true;

  // Otherwise, can it escape ?
  Board copy = impl->board;
  vector<Loc> buf;
  return copy.searchIsLadderCaptured(loc, true, buf);
}

void Game::getAreaPassAlive(vector<Stone>& out) const {
  out.clear();
  if(impl == NULL)
    return;
  const Board& board = impl->board;
  const int size = board.x_size;
  Color area[Board::MAX_ARR_SIZE];
  board.calculateArea(area, false, true, false, impl->hist.rules.multiStoneSuicideLegal);
  out.resize(size * size);
  for(int y = 0; y < size; y++) {
    for(int x = 0; x < size; x++)
      out[y * size + x] = toStone(area[Location::getLoc(x, y, size)]);
  }
}

bool Game::scoreSimple(ScoreResult& out) const {
  out = ScoreResult();
  if(impl == NULL)
    return false;
  try {
    const Board& board = impl->board;
    const int size = board.x_size;

    // Dead stones : only those inside the opponent's pass-alive regions
    vector<bool> isAlive = PlayUtils::computeAnticipatedStatusesSimple(board, impl->hist);
    Board cleared = board;
    out.isDead.assign(size * size, 0);
    for(int y = 0; y < size; y++) {
      for(int x = 0; x < size; x++) {
        Loc loc = Location::getLoc(x, y, size);
        if(board.colors[loc] != C_EMPTY && !isAlive[loc]) {
          out.isDead[y * size + x] = 1;
          cleared.setStone(loc, C_EMPTY);
        }
      }
    }

    // Score by the rules on the cleared board
    BoardHistory hist = impl->hist;
    Color area[Board::MAX_ARR_SIZE];
    hist.endAndScoreGameNow(cleared, area);
    out.scoreBlack = -hist.finalWhiteMinusBlackScore;
    out.winner = hist.winner == P_BLACK ? Stone::Black : hist.winner == P_WHITE ? Stone::White : Stone::Empty;
    out.owner.resize(size * size);
    for(int y = 0; y < size; y++) {
      for(int x = 0; x < size; x++) {
        Loc loc = Location::getLoc(x, y, size);
        Stone owner = toStone(area[loc]);
        out.owner[y * size + x] = owner;
        if(cleared.colors[loc] == C_EMPTY && owner == Stone::Black)
          out.territoryBlack++;
        else if(cleared.colors[loc] == C_EMPTY && owner == Stone::White)
          out.territoryWhite++;
      }
    }
    out.fromNeuralNet = false;
    return true;
  }
  catch(...) {
    return false;
  }
}

bool Game::scoreWithDead(const vector<uint8_t>& isDead, ScoreResult& out) const {
  out = ScoreResult();
  if(impl == NULL)
    return false;
  try {
    const Board& board = impl->board;
    const int size = board.x_size;
    if((int)isDead.size() != size * size)
      return false;

    // The dead stones removed
    Board cleared = board;
    int countDeadBlack = 0;
    int countDeadWhite = 0;
    out.isDead.assign(size * size, 0);
    for(int y = 0; y < size; y++) {
      for(int x = 0; x < size; x++) {
        Loc loc = Location::getLoc(x, y, size);
        const int index = y * size + x;
        if(board.colors[loc] == C_EMPTY || isDead[index] == 0)
          continue;
        out.isDead[index] = 1;
        if(board.colors[loc] == C_BLACK)
          countDeadBlack++;
        else
          countDeadWhite++;
        cleared.setStone(loc, C_EMPTY);
      }
    }

    // KataGo's own count on the cleared board
    BoardHistory hist = impl->hist;
    Color areaKataGo[Board::MAX_ARR_SIZE];
    hist.endAndScoreGameNow(cleared, areaKataGo);
    out.isTerritoryScoring = hist.rules.scoringRule == Rules::SCORING_TERRITORY;

    // Area : KataGo's under area scoring ; under territory scoring, each empty region bordered by one colour,
    // as the Japanese rules count it with the dame left open (KataGo's strict territory count takes any group touching a dame for a seki)
    Color area[Board::MAX_ARR_SIZE];
    if(out.isTerritoryScoring) {
      BoardHistory histArea = impl->hist;
      histArea.rules.scoringRule = Rules::SCORING_AREA;
      histArea.rules.taxRule = Rules::TAX_NONE;
      histArea.getAreaNow(cleared, area);
    }
    else
      std::copy(areaKataGo, areaKataGo + Board::MAX_ARR_SIZE, area);

    // Owner of each point : its territory and its living stones
    int countStonesBlack = 0;
    int countStonesWhite = 0;
    out.owner.resize(size * size);
    for(int y = 0; y < size; y++) {
      for(int x = 0; x < size; x++) {
        Loc loc = Location::getLoc(x, y, size);
        Stone owner = toStone(area[loc]);
        out.owner[y * size + x] = owner;
        if(cleared.colors[loc] == C_BLACK)
          countStonesBlack++;
        else if(cleared.colors[loc] == C_WHITE)
          countStonesWhite++;
        else if(owner == Stone::Black)
          out.territoryBlack++;
        else if(owner == Stone::White)
          out.territoryWhite++;
      }
    }

    // Detail : territory and prisoners, or stones and territory
    out.countBlack.territory = out.territoryBlack;
    out.countWhite.territory = out.territoryWhite;
    out.countWhite.komi = hist.rules.komi + hist.whiteHandicapBonusScore;
    if(out.isTerritoryScoring) {
      out.countBlack.prisoners = board.numWhiteCaptures;
      out.countWhite.prisoners = board.numBlackCaptures;
      out.countBlack.deadStones = countDeadWhite;
      out.countWhite.deadStones = countDeadBlack;
    }
    else {
      out.countBlack.stones = countStonesBlack;
      out.countWhite.stones = countStonesWhite;
    }
    out.countBlack.total = (float)(out.countBlack.territory + out.countBlack.stones + out.countBlack.prisoners + out.countBlack.deadStones) + out.countBlack.komi;
    out.countWhite.total = (float)(out.countWhite.territory + out.countWhite.stones + out.countWhite.prisoners + out.countWhite.deadStones) + out.countWhite.komi;

    // Result : the detail's ; KataGo's own total compared
    const float whiteMinusBlack = out.countWhite.total - out.countBlack.total;
    out.scoreBlack = -whiteMinusBlack;
    out.winner = whiteMinusBlack < 0.0f ? Stone::Black : whiteMinusBlack > 0.0f ? Stone::White : Stone::Empty;
    const float difference = whiteMinusBlack - hist.finalWhiteMinusBlackScore;
    out.isSameAsKataGo = difference > -0.01f && difference < 0.01f;
    out.fromNeuralNet = false;
    return true;
  }
  catch(...) {
    return false;
  }
}

bool Game::isTerritoryScoring() const {
  return impl != NULL && impl->hist.rules.scoringRule == Rules::SCORING_TERRITORY;
}

bool Game::loadSgf(const string& path, string& errorMsg) {
  try {
    unique_ptr<CompactSgf> sgf = CompactSgf::loadFile(path);
    if(sgf->xSize != sgf->ySize) {
      errorMsg = "Only square boards are supported";
      return false;
    }

    // Rules, setup stones and first player from the file
    Rules rulesDefault;
    const GameSetup& current = getSetup();
    if(!Rules::tryParseRulesWithoutKomi(current.rules, rulesDefault, current.komi))
      rulesDefault = Rules::getSimpleTerritory();
    Rules rules = sgf->getRulesOrFailAllowUnspecified(rulesDefault);
    Board board;
    BoardHistory hist;
    Player pla;
    sgf->setupInitialBoardAndHist(rules, board, pla, hist, BoardHistoryModes());

    GameSetup setup;
    setup.size = sgf->xSize;
    setup.rules = rules.toJsonStringNoKomi();
    setup.komi = rules.komi;
    setup.firstPlayer = pla == P_WHITE ? Stone::White : Stone::Black;
    for(int y = 0; y < setup.size; y++) {
      for(int x = 0; x < setup.size; x++) {
        Color color = board.colors[Location::getLoc(x, y, setup.size)];
        if(color == C_BLACK)
          setup.setupBlack.push_back(Move::point(x, y));
        else if(color == C_WHITE)
          setup.setupWhite.push_back(Move::point(x, y));
      }
    }

    // Moves, which must alternate
    Game loaded;
    if(!loaded.reset(setup, errorMsg))
      return false;
    for(const ::Move& move: sgf->moves) {
      if(move.pla != loaded.impl->pla) {
        errorMsg = "The moves of this SGF do not alternate";
        return false;
      }
      if(!loaded.play(fromLoc(move.loc, setup.size), errorMsg))
        return false;
    }
    *this = loaded;
    return true;
  }
  catch(const std::exception& e) {
    errorMsg = e.what();
  }
  catch(...) {
    errorMsg = "Unknown error";
  }
  return false;
}

bool Game::saveSgf(const string& path, const string& nameBlack, const string& nameWhite, string& errorMsg) const {
  if(impl == NULL) {
    errorMsg = "No game";
    return false;
  }
  try {
    ofstream out;
    if(!FileUtils::tryOpen(out, path, ios::binary)) {
      errorMsg = "Cannot write " + path;
      return false;
    }
    WriteSgf::writeSgf(out, nameBlack, nameWhite, impl->hist, vector<string>());
    out << endl;
    return true;
  }
  catch(const std::exception& e) {
    errorMsg = e.what();
  }
  return false;
}
