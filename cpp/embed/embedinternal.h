#ifndef EMBED_EMBEDINTERNAL_H_
#define EMBED_EMBEDINTERNAL_H_

// Shared by embedgame.cpp and embedengine.cpp : conversions between the public API and KataGo's types.
// Never included by the embedding application.

#include "../embed/katagoembed.h"
#include "../game/board.h"
#include "../game/boardhistory.h"
#include "../game/rules.h"

namespace katago_embed {
namespace internal {

Loc toLoc(const Move& move, int size);
Move fromLoc(Loc loc, int size);
Stone toStone(Color color);
Player toPlayer(Stone stone);

// Board, player and history of a setup, before any move.
bool buildInitial(
  const GameSetup& setup,
  const BoardHistoryModes& modes,
  Board& board,
  Player& pla,
  BoardHistory& hist,
  std::string& errorMsg
);

// One move, refused with its reason when illegal. The game ends at the first double pass (no encore).
bool applyMove(Board& board, BoardHistory& hist, Player& pla, const Move& move, std::string& errorMsg);

int countTrailingPasses(const std::vector<Move>& moves);

}  // namespace internal
}  // namespace katago_embed

#endif  // EMBED_EMBEDINTERNAL_H_
