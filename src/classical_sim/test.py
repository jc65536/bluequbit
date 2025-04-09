from dataclasses import dataclass

@dataclass
class A[T]:
    data: T

    def m[C: int](self: "A[C]") -> int:
        return self.data * 2
a1 = A(1)
a2 = A("hi")

a1.m()
a2.m()
