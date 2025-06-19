from dataclasses import dataclass
import inspect


@dataclass
class MemoryItem:
    course_name: str
    course_time: str
    description: str

    @classmethod
    def from_dict(cls, dict_input):
        return cls(
            **{
                k: v
                for k, v in dict_input.items()
                if k in inspect.signature(cls).parameters
            }
        )

    def __post_init__(self):
        # avoid text to be None
        if self.course_name is None:
            self.course_name = ""

        if self.course_time is None:
            self.course_time = ""

        if self.description is None:
            self.description = ""
