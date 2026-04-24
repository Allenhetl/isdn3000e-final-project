uint8 EMPTY=0
uint8 PLAYER_0=1
uint8 PLAYER_1=2

uint8 STATUS_WAITING=0
uint8 STATUS_PLAYING=1
uint8 STATUS_FINISHED=2
uint8 NO_PLAYER=255

string match_id
uint32 turn_index
uint8 status
uint8 current_player
uint8 winner
uint8[9] board
uint8[9] legal_actions
PieceState[] pieces
string message
