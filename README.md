# Isolation Game

This is an enhanced version of the classic Isolation board game, implemented using Pygame. Players take turns moving their pieces across a grid, blocking off squares as they go. The last player to make a valid move wins. This version includes a Player vs. AI mode with a Minimax algorithm, Player vs. Player mode, animated UI elements, and enhanced graphics.

This project was developed for the **Foundations of AI** course.

## Contributors
- Prawin Kumar S
- Malathi G

## Features

-   **Player vs. AI Mode**: Challenge an AI opponent powered by the Minimax algorithm with Alpha-Beta Pruning.
-   **Player vs. Player Mode**: Play against a friend on the same computer.
-   **Multiple Piece Types**: Choose from King, Queen, Rook, Knight, Bishop, or Pawn, each with their unique movement patterns.
-   **Animated User Interface**: Enjoy smooth transitions, particle effects, and gradient backgrounds.
-   **Dynamic Game Board**: Visually appealing grid with highlighted valid moves and blocked squares.
-   **Game Over Screen**: Clear display of the winner and game statistics.

## How to Play

### Main Menu
Upon launching the game, you'll be greeted by an animated main menu. Press any key to proceed to mode selection.

### Mode Selection
Choose between "Player vs AI" or "Player vs Player" mode.
-   **Player vs AI**: You (Player 1) will play against the computer.
-   **Player vs Player**: Two players will take turns on the same keyboard/mouse.

### Piece Selection
After selecting the game mode, you'll choose a piece type for Player 1 (and Player 2 in PvP mode). Each piece has distinct movement rules:
-   **King**: Moves one square in any direction (horizontal, vertical, or diagonal).
-   **Queen**: Moves any number of squares along any rank, file, or diagonal.
-   **Rook**: Moves any number of squares along any rank or file.
-   **Knight**: Moves in an 'L' shape (two squares in one direction, then one square perpendicularly).
-   **Bishop**: Moves any number of squares diagonally.
-   **Pawn**: Moves one square forward (Player 1 moves up, Player 2 moves down).

### Gameplay
1.  **Starting Positions**: Player 1 starts at the bottom-left corner, and Player 2 (or AI) starts at the top-right corner.
2.  **Making a Move**:
    *   Click on your piece to select it. Valid moves will be highlighted on the board.
    *   Click on a highlighted square to move your piece to that position.
    *   The square you just moved from becomes blocked and cannot be used again by either player.
3.  **Winning**: The game ends when a player cannot make any valid moves. The other player wins.

### Controls
-   **Mouse Click**: Interact with UI elements and make moves on the board.
-   **ESC**: Go back to the previous menu (from mode selection, piece selection, or reset game from playing/game over).
-   **R**: Restart the game (from game over screen).

## Installation

To run this game, you need Python and Pygame installed.

1.  **Clone the repository (or download `main.py`):**
    ```bash
    git clone https://github.com/your-username/isolation-game.git
    cd isolation-game
    ```
    (Note: Replace `https://github.com/your-username/isolation-game.git` with the actual repository URL if available, or simply ensure `main.py` is in your working directory.)

2.  **Install Pygame:**
    ```bash
    pip install pygame
    ```

## How to Run

Navigate to the directory containing `main.py` in your terminal and run:

```bash
python main.py
```

## Game Logic (AI)

The AI uses a **Minimax algorithm with Alpha-Beta Pruning** to determine its moves.
-   **Evaluation Function**: The AI evaluates game states based on:
    -   The number of valid moves available to itself versus the opponent.
    -   Future mobility (how many moves it can make from potential next positions).
    -   A bonus for blocking the opponent's best future squares.
-   **Depth**: The AI searches up to a predefined depth (default is 4 moves ahead) to find the optimal move.

## Code Structure

-   **`main.py`**: Contains the entire game logic, UI rendering, and AI implementation.
    -   `PieceType` (Enum): Defines the types of pieces.
    -   `Player` (Enum): Defines players (Human, AI, Player1, Player2).
    -   `GameMode` (Enum): Defines game modes (PvP, PvA).
    -   `Animation` class: Handles smooth movement animations.
    -   `Particle` class: Manages visual particle effects.
    -   `GameState` class: Manages the game board, piece positions, and game state.
    -   `IsolationAI` class: Implements the Minimax algorithm with Alpha-Beta Pruning.
    -   `IsolationGame` class: The main game class, handling Pygame initialization, event loops, rendering, and game flow.

## Future Enhancements

-   More sophisticated AI evaluation functions.
-   Different difficulty levels for AI.
-   Sound effects and background music.
-   Customizable board sizes.
-   Online multiplayer.
-   Improved piece graphics (e.g., 3D rendering).

## License

This project is open-source and available under the [MIT License](LICENSE).
