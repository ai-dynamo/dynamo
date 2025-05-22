use super::Storage;
use offset_allocator::{Allocation, Allocator};
use std::sync::{Arc, Mutex};

#[derive(Debug, thiserror::Error)]
pub enum ArenaError {
    #[error("Page size must be a power of 2")]
    PageSizeNotAligned,

    #[error("Allocation failed")]
    AllocationFailed,

    #[error("Failed to convert pages to u32")]
    PagesNotConvertible,
}

#[derive(Clone)]
pub struct ArenaAllocator<S: Storage> {
    storage: Arc<S>,
    allocator: Arc<Mutex<Allocator>>,
    page_size: u64,
}

pub struct ArenaBuffer<S: Storage> {
    offset: u64,
    address: u64,
    requested_size: usize,
    storage: Arc<S>,
    allocation: Allocation,
    allocator: Arc<Mutex<Allocator>>,
}

impl<S: Storage> ArenaAllocator<S> {
    pub fn new(storage: S, page_size: usize) -> Result<Self, ArenaError> {
        let storage = Arc::new(storage);

        if !page_size.is_power_of_two() {
            return Err(ArenaError::PageSizeNotAligned);
        }

        // divide storage into pages,
        // round down such that all pages are fully and any remaining bytes are discarded
        let pages = storage.size() / page_size;

        let allocator = Allocator::new(
            pages
                .try_into()
                .map_err(|_| ArenaError::PagesNotConvertible)?,
        );

        let allocator = Arc::new(Mutex::new(allocator));

        Ok(Self {
            storage,
            allocator,
            page_size: page_size as u64,
        })
    }

    pub fn allocate(&self, size: usize) -> Result<ArenaBuffer<S>, ArenaError> {
        let size = size as u64;
        let pages = (size as u64 + self.page_size - 1) / self.page_size;

        let allocation = self
            .allocator
            .lock()
            .unwrap()
            .allocate(pages.try_into().map_err(|_| ArenaError::AllocationFailed)?)
            .ok_or(ArenaError::AllocationFailed)?;

        let offset = allocation.offset as u64 * self.page_size;
        let address = self.storage.addr() + offset;

        debug_assert!(address + size <= self.storage.addr() + self.storage.size() as u64);

        Ok(ArenaBuffer {
            offset,
            address,
            requested_size: size as usize,
            allocation,
            storage: self.storage.clone(),
            allocator: self.allocator.clone(),
        })
    }
}

impl<S: Storage> std::fmt::Debug for ArenaBuffer<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ArenaBuffer {{ addr {}, size: {}, kind: {:?}, allocator: {:p}}}",
            self.address,
            self.requested_size,
            self.storage.storage_type(),
            Arc::as_ptr(&self.storage)
        )
    }
}

impl<S: Storage> ArenaBuffer<S> {
    /// Starting address of the buffer
    pub fn address(&self) -> u64 {
        self.address
    }

    /// Size of the buffer
    pub fn size(&self) -> usize {
        self.requested_size
    }
}

mod nixl {
    use super::super::nixl::*;
    use super::super::*;
    use super::*;

    impl<S: Storage> ArenaBuffer<S>
    where
        S: NixlRegisterableStorage,
    {
        pub fn to_nixl_remote_descriptor(&self) -> Result<NixlRemoteDescriptor, StorageError> {
            let agent = self.storage.nixl_agent_name();

            match agent {
                Some(agent) => {
                    // update storage with the buffer address and size
                    let storage = NixlStorage::from_storage_with_offset(
                        self.storage.as_ref(),
                        self.offset as usize,
                        self.requested_size as usize,
                    )?;

                    Ok(NixlRemoteDescriptor::new(storage, agent))
                }
                _ => Err(StorageError::NotRegisteredWithNixl),
            }
        }
    }

    impl<S: Storage> MemoryRegion for ArenaBuffer<S>
    where
        S: MemoryRegion,
    {
        unsafe fn as_ptr(&self) -> *const u8 {
            Storage::as_ptr(self.storage.as_ref())
        }

        fn size(&self) -> usize {
            Storage::size(self.storage.as_ref())
        }
    }

    impl<S: Storage> NixlDescriptor for ArenaBuffer<S>
    where
        S: NixlDescriptor,
    {
        fn mem_type(&self) -> MemType {
            NixlDescriptor::mem_type(self.storage.as_ref())
        }

        fn device_id(&self) -> u64 {
            NixlDescriptor::device_id(self.storage.as_ref())
        }
    }
}

impl<S: Storage> Drop for ArenaBuffer<S> {
    fn drop(&mut self) {
        self.allocator.lock().unwrap().free(self.allocation);
    }
}
