#ifndef EMBED_KATAGOEMBED_H_
#define EMBED_KATAGOEMBED_H_

// Public API of KataGo for an application that embeds it (Ease / Goban).
// No KataGo type crosses this header : only the standard library.
//  - Coordinates : x = column (0 = left), y = row (0 = top), point index = y * size + x.
//  - Every value is from BLACK's point of view : winrateBlack, scoreLeadBlack, ownershipBlack.
//  - Errors : methods return false and fill errorMsg ; no exception ever leaves this API.

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace katago_embed {

// Global tables of KataGo (hashes, score tables, backend). Once per process, before anything else.
bool initializeStatic(std::string& errorMsg);
void finalizeStatic();

enum class Stone : int8_t { Empty = 0, Black = 1, White = 2 };

struct Move {
  int x = -1;
  int y = -1;
  bool isPass = false;

  static Move point(int x, int y) { Move m; m.x = x; m.y = y; return m; }
  static Move pass() { Move m; m.isPass = true; return m; }
  bool isPoint() const { return !isPass && x >= 0 && y >= 0; }
  bool operator==(const Move& other) const { return x == other.x && y == other.y && isPass == other.isPass; }
};

struct GameSetup {
  int size = 19;
  std::string rules = "japanese";    // japanese, chinese, korean, aga, new-zealand, tromp-taylor, stone-scoring... or KataGo's JSON
  float komi = 6.5f;
  int handicap = 0;                  // fixed handicap stones, 0 or 2 to 9
  std::vector<Move> setupBlack;      // stones placed before the first move (an SGF's AB), after the handicap
  std::vector<Move> setupWhite;      // (an SGF's AW)
  Stone firstPlayer = Stone::Empty;  // Empty : White after a handicap, otherwise Black
};

struct BoardState {
  int size = 19;
  std::vector<Stone> stones;         // size * size
  Stone nextPlayer = Stone::Black;
  int moveNumber = 0;                // moves played since the setup, passes included
  int capturesByBlack = 0;           // white stones captured by Black
  int capturesByWhite = 0;
  Move lastMove;                     // x = -1 and isPass = false when none
  Move koForbidden;                  // x = -1 when none
  int consecutivePasses = 0;
  bool isGameOver = false;           // two consecutive passes, or finished by the rules
};

struct ScoreResult {
  float scoreBlack = 0.0f;           // > 0 : Black wins by that much
  Stone winner = Stone::Empty;
  std::vector<Stone> owner;          // owner of each point at the end : territory and living stones
  std::vector<uint8_t> isDead;       // 1 : a stone judged dead
  int territoryBlack = 0;            // empty points owned
  int territoryWhite = 0;
  bool fromNeuralNet = false;        // false : only the regions settled without any search were known
};

// Rules authority : synchronous, no neural net, cheap. Single-threaded.
class Game {
 public:
  Game();
  ~Game();
  Game(const Game& other);
  Game& operator=(const Game& other);

  bool reset(const GameSetup& setup, std::string& errorMsg);
  bool play(const Move& move, std::string& errorMsg);   // refuses an illegal move with its reason
  bool isLegal(const Move& move) const;
  void getLegalMask(std::vector<uint8_t>& mask) const;  // size * size, 1 = legal for the player to move
  bool undo();
  bool redo();                                          // the undone moves, until a new move is played
  bool canUndo() const;
  bool canRedo() const;
  void getState(BoardState& out) const;
  const GameSetup& getSetup() const;
  const std::vector<Move>& getMoves() const;            // moves played after the setup (handicap excluded)
  int getLiberties(int x, int y) const;                 // 0 on an empty point
  bool isLadderCaptured(int x, int y) const;            // a chain in atari that cannot escape
  void getAreaPassAlive(std::vector<Stone>& out) const; // regions settled without any search
  bool scoreSimple(ScoreResult& out) const;             // only what is settled without a search
  bool loadSgf(const std::string& path, std::string& errorMsg);
  bool saveSgf(const std::string& path, const std::string& nameBlack, const std::string& nameWhite, std::string& errorMsg) const;

 private:
  struct Impl;
  Impl* impl;
};

struct EngineConfig {
  std::string modelFile;
  std::string modelFileHuman;        // empty : no human-like play
  bool useGPU = true;                // onnxProvider = directml, otherwise cpu
  int indexAdapter = 0;              // onnxDeviceToUse
  int countThreads = 4;              // numSearchThreads
  std::vector<std::pair<std::string, std::string>> keysExtra;   // any other KataGo config key
};

struct PlayStyle {
  std::string profileHuman;          // "rank_10k", "preaz_5k", "rank_3d"... ; empty : full strength
  int64_t maxVisits = 400;
  double maxTime = 0.0;              // seconds, 0 : no limit
  bool allowResign = true;
};

struct Candidate {
  Move move;
  double winrateBlack = 0.5;
  double scoreLeadBlack = 0.0;
  double scoreStdev = 0.0;
  double prior = 0.0;                // the policy of the net
  int64_t visits = 0;
  int order = 0;                     // 0 = best
  std::vector<Move> pv;              // principal variation, starting with move
};

struct Analysis {
  int moveNumber = 0;
  Stone toPlay = Stone::Black;
  double winrateBlack = 0.5;
  double scoreLeadBlack = 0.0;
  int64_t visits = 0;
  std::vector<Candidate> candidates;
  std::vector<float> ownershipBlack; // size * size, -1 (White) to +1 (Black)
  std::vector<float> policy;         // size * size + 1, the pass last ; -1 on an illegal point
  bool isFinal = false;              // the last report of a search
};

struct EngineInfo {
  std::string provider;              // "directml" or "cpu" : the one actually running
  std::string errorGPU;              // why the GPU was refused, if it was
  bool hasHumanModel = false;
  double secondsLoading = 0.0;
  double visitsPerSecond = 0.0;      // measured on the last search
};

// Neural nets and search. Every method from ONE thread ; callbacks run on a KataGo thread.
class Engine {
 public:
  Engine();
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  bool initialize(const EngineConfig& config, std::string& errorMsg);   // seconds : nets loaded, graphs built
  void finalize();
  bool isInitialized() const;
  bool setPosition(const GameSetup& setup, const std::vector<Move>& moves, std::string& errorMsg);
  bool genMoveAsync(const PlayStyle& style, std::function<void(const Move& move, bool isResign, const Analysis& analysis)> onMove, std::string& errorMsg);
  bool analyzeAsync(int64_t maxVisits, double periodSeconds, int countCandidates, std::function<void(const Analysis&)> onAnalysis, std::string& errorMsg);
  bool evaluateRaw(Analysis& out, std::string& errorMsg);                    // the net alone, no search : instant
  bool scoreFinal(ScoreResult& out, int64_t visits, std::string& errorMsg);  // dead stones from the ownership of a search
  void stop();                                                               // stops a search and waits for it
  void setLogCallback(std::function<void(const std::string& line)> onLog);   // before initialize
  EngineInfo getInfo() const;

  // DirectML usable ? Creates a session on a one-node graph with the DirectML provider.
  static bool probeDirectML(int indexAdapter, std::string& errorMsg);

  // A model file held in memory (assets packed in the executable) : initialize reads it there when a config names this path.
  // The bytes must stay valid while an engine may load it.
  static void registerMemoryFile(const std::string& path, const void* data, size_t size);

 private:
  struct Impl;
  Impl* impl;
};

}  // namespace katago_embed

#endif  // EMBED_KATAGOEMBED_H_
