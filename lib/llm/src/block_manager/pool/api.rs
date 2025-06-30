

pub trait BlockPool {
    fn allocate_blocks(&self, count: usize) -> Result<Vec<MutableBlock>, BlockPoolError>;
    fn register_blocks(&self, blocks: Vec<MutableBlock>) -> Result<Vec<ImmutableBlock>, BlockPoolError>;
    fn match_sequence_hashes(&self, sequence_hashes: Vec<SequenceHash>) -> Result<Vec<ImmutableBlock>, BlockPoolError>;
}




