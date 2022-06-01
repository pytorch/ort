class ProviderOptions:
    def __init__(
        self,
        provider="",
    ):
        self._provider = provider
        
    @property
    def provider(self):
        return self._provider
        
class OpenVINOProviderOptions(ProviderOptions):
    def __init__(
        self,
        provider="",
        backend="CPU",
        precision="FP32",
    ):
        super().__init__(provider)
        self._backend = backend
        self._precision = precision

    @property
    def backend(self):
        return self._backend

    @property
    def precision(self):
        return self._precision
