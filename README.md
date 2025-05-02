# Multistage-Morris-AI

## Overview

Multistage-Morris-AI is an advanced Python-based project dedicated to implementing artificial intelligence techniques to play and solve the Morris board game. This repository explores strategic decision-making, heuristic evaluations, and optimization algorithms to simulate intelligent gameplay. The multi-stage approach ensures modularity and scalability, making it an excellent resource for AI researchers, developers, and enthusiasts interested in game theory and artificial intelligence.

## Features

- **Multi-Stage Architecture**: Divides the AI decision-making process into multiple stages for clarity and modularity.
- **Heuristic Evaluation**: Implements evaluation functions to assess board states and predict optimal strategies.
- **Optimization Algorithms**: Leverages advanced algorithms to enhance decision-making and gameplay performance.
- **Customizability**: Modular design allows for easy extension and adaptation of components for experimentation.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Game Rules](#game-rules)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

To get started with Multistage-Morris-AI, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/jakubkalinski0/Multistage-Morris-AI.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Multistage-Morris-AI
   ```
3. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate # On Windows: venv\Scripts\activate
   ```
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the AI and simulate a game:

1. Execute the main script:
   ```bash
   python main.py
   ```
2. Follow the on-screen instructions to configure the game and AI settings.

For more advanced usage, refer to the documentation or the comments in the source code.

## Game Rules

Morris is a traditional two-player strategy board game. Here is a brief overview of the rules:

1. Each player places their pieces alternately on the board.
2. The objective is to form "mills" (three pieces in a row), allowing the player to remove an opponent's piece.
3. Once all pieces are placed, players take turns moving their pieces to adjacent spots.
4. A player wins by reducing the opponent to two pieces or blocking all possible moves.

For more details, refer to the official rules of Morris.

## Project Structure

```
## Project Structure

├── main.py                # Entry point of the application
├── ai/
│   ├── strategy.py        # Core AI decision-making logic
│   ├── heuristics.py      # Heuristic evaluation functions
│   └── optimization.py    # Optimization algorithms
├── tests/
│   ├── test_strategy.py   # Unit tests for strategy logic
│   └── test_heuristics.py # Unit tests for heuristics
├── README.md              # Project documentation
├── requirements.txt       # Dependencies for the project
└── LICENSE                # License information
```

## Contributing

We welcome contributions to enhance the functionality and performance of Multistage-Morris-AI. To contribute:

1. Fork the repository on GitHub.
2. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes and push the branch:
   ```bash
   git commit -m "Add new feature"
   git push origin feature-name
   ```
4. Open a pull request on the main repository.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Contact

For questions or feedback, please contact:
- **Authors**: Jakub Kaliński, Kacper Feliks
- **GitHub Profiles**: [@jakubkalinski0](https://github.com/jakubkalinski0) [@Kacperon](https://github.com/Kacperon)
