from types import MappingProxyType

CHOICES = MappingProxyType({
    "noise_schedule": ('linear', 'cosine', 'sqrt', 'trunc_cos', 'trunc_lin', 'pw_lin'),
    "schedule_sampler": ('uniform', 'lossaware', 'fixstep')
})
