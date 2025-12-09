import abc

class Tool(abc.ABC):
    name: str
    description: str

    @abc.abstractmethod
    def __call__(self, **kwargs) -> str:
        pass
