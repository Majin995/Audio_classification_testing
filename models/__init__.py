from .hydro_conformer import HydroConformer
from .hydro_net import HydroNet
from .hydro_resnet import HydroResNet
from .hydro_leaf import HydroLEAF
from .hydro_s4 import HydroS4
from .hydro_ssamba import HydroSSAMBA
from .hydro_spiking_leaf import HydroSpikingLEAF
from .hydro_panns import HydroPANNs
from .hydro_xlsr import HydroXLSR
from .hydro_fusion import HydroFusion
from .hydro_ensemble import HydroEnsemble
from .hydro_binary_ensemble import HydroBinaryEnsemble
from .hydro_lofar_resnet import HydroLofarResNet
from .hydro_eat import HydroEAT
from .hydro_mae import HydroMAE
from .hydro_i2hofi import I2HOFI
from .hydro_dart_mt import HydroDARTMT
from .hydro_cnn1d import HydroCNN1D
from .hydro_omni_resnet import AcousticOmniResNet
from .hydro_uast3d import HydroUAST3D
# ── New UATR paradigms ─────────────────────────────────────────────────────
from .hydro_catfish import HydroCATFISH
from .hydro_alsi import HydroALSI
from .hydro_dcn import HydroDCN
from .hydro_bahtnet import HydroBAHTNet
from .hydro_sscp_mobile import HydroSSCPMobile
from .hydro_precise import HydroPrecise
from .hydro_precise_v2 import HydroPreciseV2
from .hydro_hydra import HydroHydra
from .hydro_complete import HydroComplete
from .hydro_wave1d import HydroWave1D
from .hydro_wave_scattering import HydroWaveScattering
from .super_model import SuperModel1D
# ── SSL pretraining + Audio-LLM hybrid ─────────────────────────────────────
from .hydro_barlow_twins import HydroBarlowTwins
from .hydro_audio_llm    import HydroAudioLLM
# ── Quantum / Hybrid models ────────────────────────────────────────────────
# Optional: depend on torchquantum (declared in requirements.txt). Skip
# gracefully when the dep isn't installed so non-quantum training paths still
# load.
try:
    from .hydro_cnnlstm_qc       import HydroCNNLSTMQC
    from .hydro_quantum_transfer import HydroQuantumTransfer
    from .hydro_vqc_features     import HydroVQCFeatures
    _QUANTUM_AVAILABLE = True
except ImportError:
    _QUANTUM_AVAILABLE = False

__all__ = [
    # ── Original models ───────────────────────────────────────────────────
    "HydroConformer", "HydroNet", "HydroResNet", "HydroLEAF", "HydroS4",
    "HydroSSAMBA", "HydroSpikingLEAF", "HydroPANNs", "HydroXLSR",
    "HydroFusion", "HydroEnsemble", "HydroBinaryEnsemble",
    "HydroLofarResNet", "HydroEAT", "HydroMAE", "I2HOFI",
    "HydroDARTMT", "HydroCNN1D", "AcousticOmniResNet", "HydroUAST3D",
    # ── New UATR paradigms ────────────────────────────────────────────────
    "HydroCATFISH", "HydroALSI", "HydroDCN", "HydroBAHTNet", "HydroSSCPMobile",
    "HydroPrecise", "HydroPreciseV2",
    "HydroHydra", "HydroComplete",
    "HydroWave1D", "HydroWaveScattering",
    # ── Super multi-stream model ──────────────────────────────────────────
    "SuperModel1D",
    # ── SSL pretraining + Audio-LLM hybrid ────────────────────────────────
    "HydroBarlowTwins", "HydroAudioLLM",
]
if _QUANTUM_AVAILABLE:
    __all__ += ["HydroCNNLSTMQC", "HydroQuantumTransfer", "HydroVQCFeatures"]
