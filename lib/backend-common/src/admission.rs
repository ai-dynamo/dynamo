// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Optional controller permission, independent of engine health and operator pause.
//! The controller retains durable intent; a new worker starts closed with a fresh lifetime.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use serde::{Deserialize, Serialize};
use tokio::sync::watch;

use crate::{BackendError, DynamoError, ErrorType};

/// Engine-observed identity and committed topology, supplied by the backend's observer.
/// A controller grant cannot establish this binding or substitute for engine observation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdmissionBinding {
    pub engine_instance_id: String,
    pub topology_generation: i64,
}

/// Absolute permission for one worker lifetime and exact engine/topology binding.
/// Granting permission attests that the controller observed commit and serving verification.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdmissionTarget {
    pub control_revision: i64,
    pub worker_instance_id: String,
    pub group_id: String,
    pub native_member_id: String,
    pub binding: AdmissionBinding,
    pub admitted: bool,
}

/// Accepted intent is distinct from the registration/withdrawal the worker has acknowledged.
/// Neither discovery withdrawal nor this status is an engine drain proof.
#[derive(Clone, Debug, Serialize)]
pub struct AdmissionStatus {
    pub worker_instance_id: String,
    pub group_id: String,
    pub native_member_id: String,
    pub binding: Option<AdmissionBinding>,
    pub engine_ready: bool,
    pub accepted: Option<AdmissionTarget>,
    pub observed: Option<AdmissionTarget>,
}

struct AdmissionState {
    status: watch::Sender<AdmissionStatus>,
    is_admitted: AtomicBool,
}

/// Shared by the existing worker, request adapter, control routes and backend observer.
/// No Kubernetes client, collective operation journal or engine recovery loop lives here.
#[derive(Clone)]
pub struct WorkerAdmission {
    state: Arc<AdmissionState>,
}

impl WorkerAdmission {
    pub fn new(group_id: String, native_member_id: String) -> Result<Self, DynamoError> {
        validate_id(&group_id)?;
        validate_id(&native_member_id)?;
        Ok(Self {
            state: Arc::new(AdmissionState {
                status: watch::channel(AdmissionStatus {
                    worker_instance_id: uuid::Uuid::new_v4().to_string(),
                    group_id,
                    native_member_id,
                    binding: None,
                    engine_ready: false,
                    accepted: None,
                    observed: None,
                })
                .0,
                is_admitted: AtomicBool::new(false),
            }),
        })
    }

    pub fn observe(&self) -> AdmissionStatus {
        self.state.status.borrow().clone()
    }

    /// Replaying the exact accepted target wakes reconciliation without repeating engine work.
    /// Rejection never overwrites the last accepted target.
    pub fn apply(&self, target: AdmissionTarget) -> Result<(), DynamoError> {
        validate_id(&target.binding.engine_instance_id)?;
        if target.control_revision <= 0 || target.binding.topology_generation <= 0 {
            return Err(invalid(
                "admission revision and topology generation must be positive",
            ));
        }
        let mut result = Ok(());
        self.state.status.send_if_modified(|status| {
            if target.worker_instance_id != status.worker_instance_id
                || target.group_id != status.group_id
                || target.native_member_id != status.native_member_id
            {
                result = Err(invalid(
                    "admission target belongs to a different worker or member",
                ));
                return false;
            }
            if let Some(accepted) = &status.accepted {
                if target.control_revision < accepted.control_revision
                    || (target.control_revision == accepted.control_revision && target != *accepted)
                {
                    result = Err(invalid(
                        "admission revision is stale or has a different payload",
                    ));
                    return false;
                }
                if target == *accepted {
                    return true;
                }
            }
            if target.admitted && status.binding.as_ref() != Some(&target.binding) {
                result = Err(DynamoError::builder()
                    .error_type(ErrorType::Backend(BackendError::CannotConnect))
                    .message("matching engine and topology binding is not yet observed")
                    .build());
                return false;
            }
            self.state.is_admitted.store(false, Ordering::Release);
            status.observed = None;
            status.accepted = Some(target);
            true
        });
        result
    }

    /// Called only by a trusted backend observer, never by the controller-facing target route.
    pub fn bind(&self, binding: AdmissionBinding) -> Result<(), DynamoError> {
        validate_id(&binding.engine_instance_id)?;
        if binding.topology_generation <= 0 {
            return Err(invalid("observed topology generation must be positive"));
        }
        self.state.status.send_if_modified(|status| {
            if status.binding.as_ref() == Some(&binding) {
                return false;
            }
            self.state.is_admitted.store(false, Ordering::Release);
            status.observed = None;
            if status
                .binding
                .as_ref()
                .is_some_and(|previous| previous.engine_instance_id != binding.engine_instance_id)
            {
                status.engine_ready = false;
            }
            status.binding = Some(binding);
            true
        });
        Ok(())
    }

    pub fn invalidate(&self) {
        self.state.status.send_modify(|status| {
            self.state.is_admitted.store(false, Ordering::Release);
            status.binding = None;
            status.observed = None;
            status.engine_ready = false;
        });
    }

    pub fn is_admitted(&self) -> bool {
        self.state.is_admitted.load(Ordering::Acquire)
    }

    pub(crate) fn ensure_admitted(&self) -> Result<(), DynamoError> {
        if self.is_admitted() {
            return Ok(());
        }
        Err(DynamoError::builder()
            .error_type(ErrorType::Backend(BackendError::CannotConnect))
            .message("worker is not admitted to the Engine Group")
            .build())
    }

    pub(crate) fn subscribe(&self) -> watch::Receiver<AdmissionStatus> {
        self.state.status.subscribe()
    }

    pub(crate) fn notify(&self) {
        self.state.status.send_modify(|_| {});
    }

    pub(crate) fn set_engine_ready(&self, ready: bool) {
        self.state.status.send_if_modified(|status| {
            if status.engine_ready == ready {
                return false;
            }
            if !ready {
                self.state.is_admitted.store(false, Ordering::Release);
                status.observed = None;
            }
            status.engine_ready = ready;
            true
        });
    }

    pub(crate) fn registration_target(&self) -> Option<AdmissionTarget> {
        let status = self.state.status.borrow();
        status
            .accepted
            .as_ref()
            .filter(|target| {
                status.engine_ready
                    && target.admitted
                    && status.binding.as_ref() == Some(&target.binding)
            })
            .cloned()
    }

    pub(crate) fn clear_registration(&self) {
        self.state.status.send_if_modified(|status| {
            self.state.is_admitted.store(false, Ordering::Release);
            if status
                .observed
                .as_ref()
                .is_some_and(|target| target.admitted)
            {
                status.observed = None;
                true
            } else {
                false
            }
        });
    }

    pub(crate) fn confirm_registration(&self, target: &AdmissionTarget) -> bool {
        let mut confirmed = false;
        self.state.status.send_if_modified(|status| {
            confirmed = status.engine_ready
                && status.accepted.as_ref() == Some(target)
                && target.admitted
                && status.binding.as_ref() == Some(&target.binding);
            if !confirmed {
                return false;
            }
            self.state.is_admitted.store(true, Ordering::Release);
            if status.observed.as_ref() == Some(target) {
                return false;
            }
            status.observed = Some(target.clone());
            true
        });
        confirmed
    }

    pub(crate) fn confirm_withdrawal(&self, target: &AdmissionTarget) {
        self.state.status.send_if_modified(|status| {
            if target.admitted
                || status.accepted.as_ref() != Some(target)
                || status.observed.as_ref() == Some(target)
            {
                return false;
            }
            self.state.is_admitted.store(false, Ordering::Release);
            status.observed = Some(target.clone());
            true
        });
    }
}

fn validate_id(id: &str) -> Result<(), DynamoError> {
    if id.trim().is_empty() || id.len() > 256 {
        return Err(invalid(
            "admission identities must contain 1–256 nonblank bytes",
        ));
    }
    Ok(())
}

fn invalid(message: &str) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Backend(BackendError::InvalidArgument))
        .message(message)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target(admission: &WorkerAdmission) -> AdmissionTarget {
        let status = admission.observe();
        AdmissionTarget {
            control_revision: 1,
            worker_instance_id: status.worker_instance_id,
            group_id: status.group_id,
            native_member_id: status.native_member_id,
            binding: AdmissionBinding {
                engine_instance_id: "engine-1".into(),
                topology_generation: 1,
            },
            admitted: true,
        }
    }

    #[test]
    fn permission_requires_observation_readiness_and_confirmed_registration() {
        let admission = WorkerAdmission::new("group-1".into(), "dp-1".into()).unwrap();
        let grant = target(&admission);
        assert!(!admission.is_admitted());
        assert!(
            admission.apply(grant.clone()).is_err(),
            "a controller cannot invent engine identity"
        );
        admission.bind(grant.binding.clone()).unwrap();
        admission.apply(grant.clone()).unwrap();
        assert!(
            admission.registration_target().is_none(),
            "permission is not engine readiness"
        );
        assert!(!admission.confirm_registration(&grant));
        admission.set_engine_ready(true);
        assert_eq!(admission.registration_target(), Some(grant.clone()));
        assert!(!admission.is_admitted(), "acceptance is not registration");
        assert!(admission.confirm_registration(&grant));
        assert!(admission.is_admitted());
    }

    #[test]
    fn rejected_targets_preserve_accepted_intent_and_revocation_fences_late_registration() {
        let admission = WorkerAdmission::new("group-1".into(), "dp-1".into()).unwrap();
        let grant = target(&admission);
        admission.bind(grant.binding.clone()).unwrap();
        admission.set_engine_ready(true);
        admission.apply(grant.clone()).unwrap();
        assert!(admission.confirm_registration(&grant));
        let mut invalid = grant.clone();
        invalid.admitted = false;
        assert!(
            admission.apply(invalid).is_err(),
            "one revision has one exact payload"
        );
        assert_eq!(admission.observe().accepted, Some(grant.clone()));
        assert!(admission.is_admitted());
        admission.apply(grant.clone()).unwrap();
        assert!(
            admission.is_admitted(),
            "an identical replay does not briefly revoke permission"
        );
        let mut withdrawal = grant.clone();
        withdrawal.control_revision = 2;
        withdrawal.admitted = false;
        admission.apply(withdrawal.clone()).unwrap();
        assert!(
            !admission.is_admitted(),
            "revocation closes local admission before discovery I/O"
        );
        assert!(
            !admission.confirm_registration(&grant),
            "late completion cannot acknowledge a superseded grant"
        );
        assert_eq!(admission.observe().accepted, Some(withdrawal));
        assert!(admission.apply(grant).is_err());
    }

    #[test]
    fn changed_engine_topology_or_worker_lifetime_cannot_inherit_permission() {
        let admission = WorkerAdmission::new("group-1".into(), "dp-1".into()).unwrap();
        let grant = target(&admission);
        admission.bind(grant.binding.clone()).unwrap();
        admission.set_engine_ready(true);
        admission.apply(grant.clone()).unwrap();
        assert!(admission.confirm_registration(&grant));
        admission
            .bind(AdmissionBinding {
                engine_instance_id: "engine-1".into(),
                topology_generation: 2,
            })
            .unwrap();
        assert!(!admission.is_admitted());
        assert!(!admission.confirm_registration(&grant));
        admission.invalidate();
        assert!(admission.observe().binding.is_none());
        assert!(admission.registration_target().is_none());
        admission
            .bind(AdmissionBinding {
                engine_instance_id: "engine-2".into(),
                topology_generation: 2,
            })
            .unwrap();
        admission.set_engine_ready(true);
        assert!(admission.registration_target().is_none());
        let restarted = WorkerAdmission::new("group-1".into(), "dp-1".into()).unwrap();
        restarted.bind(grant.binding.clone()).unwrap();
        restarted.set_engine_ready(true);
        assert!(restarted.apply(grant).is_err());
        assert!(!restarted.is_admitted());
    }
}
