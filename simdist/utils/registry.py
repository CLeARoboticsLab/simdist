from typing import Callable, Generic, TypeVar


T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str):
        self.kind = kind
        self._registry: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        def decorator(cls: type[T]) -> type[T]:
            if name in self._registry:
                raise ValueError(
                    f"{self.kind} '{name}' already registered. "
                    f"Registered: {self.names()}"
                )
            self._registry[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> type[T]:
        if name not in self._registry:
            raise ValueError(
                f"{self.kind} '{name}' not found. "
                f"Registered: {self.names()}"
            )
        return self._registry[name]

    def create(self, name: str, *args, **kwargs) -> T:
        return self.get(name)(*args, **kwargs)

    def names(self) -> list[str]:
        return list(self._registry.keys())
