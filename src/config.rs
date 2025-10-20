use serde::{Deserialize, Serialize};
use crate::bindings::{Context, QE};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Camera {
    pub name: String,
    pub qe_red: QuantumEfficiency,
    pub qe_green: QuantumEfficiency,
    pub qe_blue: QuantumEfficiency
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct QuantumEfficiency {
    #[serde(rename = "ha")]
    pub hydrogen_alpha: f64,
    #[serde(rename = "oiii")]
    pub oxygen_iii: f64
}

impl QuantumEfficiency {
    pub fn as_gpu_qe(self, context: &'_ Context) -> QE<'_> {
        QE::new(context, self.hydrogen_alpha, self.oxygen_iii).unwrap()
    }
}