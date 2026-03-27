from .sae_text_steering import SAETextSteering
from .gcg import GCGSteering
from .refusal_only import RefusalOnlySteering
from .random_feature import RandomFeatureSteering
from .refusal_dir import RefusalDirActadd, RefusalDirAblation
from .angular_steering import AngularSteering


def get_steering_method(config):
    methods = {
        'sae_text_steering':         SAETextSteering,
        'sae_text_steering_greedy':    SAETextSteering,
        'sae_text_steering_sampling':  SAETextSteering,
        'sae_text_steering_refdir': SAETextSteering,
        'sae_refdir_prehook':         SAETextSteering,
        'sae_refdir_fwdhook':         SAETextSteering,
        'sae_refdir_prehook_greedy':       SAETextSteering,
        'sae_refdir_prehook_nosub':        SAETextSteering,
        'sae_refdir_fwdhook_greedy':       SAETextSteering,
        'sae_refdir_fwdhook_greedy_layer16': SAETextSteering,
        'sae_olddir_fwdhook_greedy':       SAETextSteering,
        'sae_olddir_prehook_greedy':       SAETextSteering,
        'sae_olddir_prehook_l15_greedy':   SAETextSteering,
        'sae_olddir_fwdhook_l15_greedy':   SAETextSteering,
        'sae_normalized_nosub':            SAETextSteering,
        'sae_normalized_toolsdir':         SAETextSteering,
        'sae_combined_maxtext':            SAETextSteering,
        'refusal_dir_actadd_greedy':       RefusalDirActadd,
        'angular_refusal_greedy':          AngularSteering,
        'angular_refusal_90_greedy':       AngularSteering,
        'angular_refusal_180_toolsdir':         AngularSteering,
        'angular_refusal_90_toolsdir':          AngularSteering,
        'angular_refusal_180_toolsdir_prehook': AngularSteering,
        'angular_sae_refusal_greedy':          AngularSteering,
        'angular_sae_only_greedy':             AngularSteering,
        'angular_sae_maxtext_refusal_90':      AngularSteering,
        'angular_sae_maxtext_refusal_120':     AngularSteering,
        'angular_sae_maxtext_refusal_150':     AngularSteering,
        'angular_sae_maxtext_refusal_180':              AngularSteering,
        'angular_sae_150_refusal_180_sae_first':        AngularSteering,
        'angular_sae_150_refusal_180_refusal_first':    AngularSteering,
        'angular_sae_decoder_150':                      AngularSteering,
        'gcg':                 GCGSteering,
        'refusal_only':        RefusalOnlySteering,
        'random_feature':      RandomFeatureSteering,
        'refusal_dir_actadd':  RefusalDirActadd,
        'refusal_dir_ablation': RefusalDirAblation,
    }
    return methods[config.method.name](config)
