/// Macro to generate WriteTo implementations for specific storage type combinations
///
/// This macro generates the boilerplate for implementing WriteTo for a coordinator
/// when the source and destination storage types have a defined strategy.
#[macro_export]
macro_rules! impl_write_to_strategy {
    (
        $coordinator:ty,
        $src_storage:ty => $dst_storage:ty,
        $src_locality:ty => $dst_locality:ty,
        $executor:path
    ) => {
        impl<M: BlockMetadata> WriteTo<$src_storage, $src_locality, $dst_storage, $dst_locality, M>
            for $coordinator
        {
            fn write_to(
                &self,
                src: &[&Block<$src_storage, $src_locality, M>],
                dst: &mut [&mut Block<$dst_storage, $dst_locality, M>],
                notify: bool,
                ctx: Arc<TransferContext>,
            ) -> Result<Option<oneshot::Receiver<()>>, TransferError> {
                if src.len() != dst.len() {
                    return Err(TransferError::CountMismatch(src.len(), dst.len()));
                }

                let (tx, rx) = if notify {
                    let (t, r) = oneshot::channel();
                    (Some(t), Some(r))
                } else {
                    (None, None)
                };

                // Execute the transfer using the specified executor
                $executor(src, dst, &ctx)?;

                // Notify completion
                if let Some(tx) = tx {
                    let _ = tx.send(());
                }

                Ok(rx)
            }
        }
    };
}

/// Macro to generate multiple WriteTo implementations for a coordinator
///
/// This macro takes a coordinator type and a list of storage combinations,
/// automatically generating implementations for each combination.
#[macro_export]
macro_rules! impl_coordinator_strategies {
    (
        $coordinator:ty,
        $(
            $src_storage:ty => $dst_storage:ty,
            $src_locality:ty => $dst_locality:ty,
            $executor:path
        ),* $(,)?
    ) => {
        $(
            impl_write_to_strategy!(
                $coordinator,
                $src_storage => $dst_storage,
                $src_locality => $dst_locality,
                $executor
            );
        )*
    };
}

/// Macro to generate WriteToBlocks implementations for common storage combinations
///
/// This generates the extension trait implementations that allow calling
/// `vec_of_immutable_blocks.write_to(vec_of_mutable_blocks, notify, ctx)`
#[macro_export]
macro_rules! impl_write_to_blocks {
    (
        $coordinator:ty,
        $(
            $src_storage:ty => $dst_storage:ty,
            $src_locality:ty => $dst_locality:ty
        ),* $(,)?
    ) => {
        $(
            impl<M: BlockMetadata> WriteToBlocks<$dst_storage, $dst_locality, M>
                for Vec<ImmutableBlock<$src_storage, $src_locality, M>>
            {
                fn write_to(
                    &self,
                    dst: &mut [&mut Block<$dst_storage, $dst_locality, M>],
                    notify: bool,
                    ctx: Arc<TransferContext>,
                ) -> Result<Option<oneshot::Receiver<()>>, TransferError> {
                    let coordinator = <$coordinator>::default();
                    coordinator.write_to(self, dst, notify, ctx)
                }
            }
        )*
    };
}

/// Helper macro to reduce repetition when implementing for Local locality
#[macro_export]
macro_rules! impl_local_transfers {
    (
        $coordinator:ty,
        $(
            $src_storage:ty => $dst_storage:ty,
            $executor:path
        ),* $(,)?
    ) => {
        impl_coordinator_strategies!(
            $coordinator,
            $(
                $src_storage => $dst_storage,
                locality::Local => locality::Local,
                $executor
            ),*
        );

        impl_write_to_blocks!(
            $coordinator,
            $(
                $src_storage => $dst_storage,
                locality::Local => locality::Local
            ),*
        );
    };
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::block_manager::{DeviceStorage, PinnedStorage};

    // Example usage of the macros (this would normally be in executors.rs)
    #[derive(Default)]
    struct TestCoordinator;

    fn test_executor<M: BlockMetadata>(
        _src: &[&Block<DeviceStorage, locality::Local, M>],
        _dst: &mut [&mut Block<PinnedStorage, locality::Local, M>],
        _ctx: &TransferContext,
    ) -> Result<(), TransferError> {
        Ok(())
    }

    // This demonstrates how the macro would be used
    impl_write_to_strategy!(
        TestCoordinator,
        DeviceStorage => PinnedStorage,
        locality::Local => locality::Local,
        test_executor
    );

    #[test]
    fn test_macro_compilation() {
        // Just testing that the macro expands correctly
        let _coordinator = TestCoordinator::default();
    }
}
