class Position:
    """Position representation on the board."""
    def __init__(self, id: int):
        self.id = id
        self.connections = set()

    def add_connection(self, position_id: int):
        """Adds a connection to the position."""
        self.connections.add(position_id)

    def is_connected(self, position_id: int):
        """Checks if the position is connected to specified position."""
        return position_id in self.connections

    def __eq__(self, other):
        if not isinstance(other, Position):
            return False
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)