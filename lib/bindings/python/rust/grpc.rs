use pyo3::prelude::*;


pub struct KserveGrpcService {
    inner: kserve::KserveService,
}

#[pymethods]
impl KserveGrpcService {
    #[new]
    #[pyo3(signature = (port=None, host=None))]
    pub fb new(port: )