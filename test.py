class OffloadFilter:
    def __init__(self):
        ...

    def should_offload(self, sequence_hash: int) -> bool:
        print("SHOULD OFFLOAD CALLED!")
        return True