from functools import total_ordering
from typing import List, Optional, Tuple, Generic, TypeVar, Dict, Union


class ParamEntry:
    """
    Represents the type for a parameter or return of an API entry.
    """

    UNKNOWN = "UNKNOWN"
    ANY = "ANY"

    def __init__(self, type_name: str):
        """
        Instantiates an instance of a parameter type entry.

        :param type_name: the type name. Use 'UNKNOWN' or 'ANY' as necessary.
        """
        if type_name.endswith("..."):
            self.type_name = type_name.rstrip("...")
            self.varargs = True
        else:
            self.type_name = type_name
            self.varargs = False

    def is_unknown(self) -> bool:
        """
        :return: true if the type refers to an unknown or undefined type.
        """
        return ParamEntry.UNKNOWN.casefold() == self.type_name.casefold()

    def is_any(self) -> bool:
        """
        :return: true if the type refers to an "any" type.
        """
        return ParamEntry.ANY.casefold() == self.type_name.casefold()

    def __eq__(self, o):
        if not isinstance(o, ParamEntry):
            return False
        else:
            return self.type_name == o.type_name

    def __str__(self):
        return self.type_name

    def __repr__(self):
        return str(self.__dict__)

    def __hash__(self):
        return hash(self.__dict__)


class CallEntry:
    """
    Represents a public facing procedure that can be used from a given type.
    """

    def __init__(self,
                 name: str,
                 parameter_entries: List[ParamEntry] = tuple([]),
                 return_entries: List[ParamEntry] = tuple([])):
        """
        Instantiates an instance of a public procedure (API) entry.

        :param name: The name of the public facing procedure name.
        :param parameter_entries: An ordered list of the parameter entries or arguments of the API.
        :param return_entries: The parameter entry type.
        """
        self.name = name
        self._parameter_entries = tuple(parameter_entries)
        self.parameter_count = len(self._parameter_entries)
        self.return_entries = return_entries

    def get_parameter_at(self, index: int) -> Optional[ParamEntry]:
        """
        Returns the parameter entry found at that index.

        :param index: the index where the entry lies.
        :return: the optional value containing the entry if the index reflects a valid index position, empty if
            otherwise.
        """
        if 0 <= index < len(self._parameter_entries):
            return self._parameter_entries[index]
        else:
            return None

    @staticmethod
    def from_json(json_dct):

        def handle_return_type(return_json):
            if isinstance(return_json, str):
                return [ParamEntry(json_dct['returnType'])]
            else:
                return list(map(lambda x: ParamEntry(x), json_dct['returnType']))

        return CallEntry(
            json_dct['callName'],
            list(map(lambda x: ParamEntry(x), json_dct['paramTypes'])),
            handle_return_type(json_dct['returnType']))

    def to_json(self) -> dict:
        return {
            'callName': self.name,
            'paramTypes': [x.type_name for x in self._parameter_entries],
            'returnType': [x.type_name for x in self.return_entries],
        }

    def __str__(self):
        return "{}({}):{}".format(
            self.name,
            ",".join(map(str, self._parameter_entries)),
            self.return_entries
        )

    def __repr__(self):
        return str(self.__dict__)

    def __eq__(self, o):
        if not isinstance(o, CallEntry):
            return False
        else:
            return self.name == o.name and \
                self._parameter_entries == o._parameter_entries and \
                self.return_entries == o.return_entries

    def __hash__(self):
        return hash(self.__dict__.values())


class TypeEntry:
    """
    Represents a type that can be instantiated directory or referred to statically and
    exposes a set of public facing APIs.
    """

    def __init__(self, name: str, api_entries: List[CallEntry] = None):
        """
        Instantiates an instance of a type entry.

        :param name: The type declaration name.
        :param api_entries: The set of public facing procedures exposed by this type.
        """
        self.name = name
        if api_entries is None:
            self.api_entries = tuple([])
        else:
            self.api_entries = tuple(api_entries)

    def __eq__(self, o):
        if not isinstance(o, TypeEntry):
            return False
        else:
            return self.name == o.name and \
                self.api_entries == o.api_entries

    def __str__(self):
        return "TypeEntry({})".format(self.name)

    def __repr__(self):
        return str(self.__dict__)

    def __hash__(self):
        return hash(self.__dict__)

    def to_json(self) -> dict:
        return {
            'name': self.name,
            'apiEntries': [a.to_json() for a in self.api_entries]
        }

    @staticmethod
    def from_json(json_dct: dict):
        return TypeEntry(
            json_dct['name'],
            list(map(lambda x: CallEntry.from_json(x), json_dct['apiEntries'])),
        )


T = TypeVar("T")


@total_ordering
class WeightedPair(Generic[T]):
    """
    A pair with a single weight. The comparison of two pairs is based on the weight.
    """

    def __init__(self, left: T, right: T, weight: float):
        """
        Instantiates a weighted pair.
        :param left: The left-hand element.
        :param right: The right-hand element.
        :param weight: The weight of the pair.
        """
        self.left = left
        self.right = right
        self.weight = weight

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, WeightedPair):
            return False
        else:
            return self.weight == o.weight

    def __le__(self, o: object) -> bool:
        if not isinstance(o, WeightedPair):
            return False
        else:
            return self.weight <= o.weight

    def __hash__(self):
        return hash(self.__dict__)


@total_ordering
class WeightedSingleton(Generic[T]):
    """
    A simple class representing a weighted objected. The comparison of two singletons is based on the weight.
    """

    def __init__(self, obj: T, weight: float):
        """
        Instantiates a weighted singleton.

        :param obj: The weighted object.
        :param weight: The weight of the object.
        """
        self.obj = obj
        self.weight = weight

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, WeightedSingleton):
            return False
        else:
            return self.weight == o.weight

    def __le__(self, o: object) -> bool:
        if not isinstance(o, WeightedSingleton):
            return False
        else:
            return self.weight <= o.weight

    def __hash__(self):
        return hash(self.__dict__)

    def __str__(self):
        return "({}, {})".format(self.obj, self.weight)

    def __repr__(self):
        return str(self.__dict__)


class DefComponent:
    """
    Represents a component that carries data. This could be an identifier of a variable or method and supplementary type
    information, if available.
    """

    def __init__(self, name: str, type_full_name: str = "ANY", literal: bool = False):
        """
        Instantiates a definition component.

        :param name: the name of the object or method call.
        :param type_full_name: the type full name.
        :param literal: if this object represents a literal or not.
        """
        self.name = name
        self.type_full_name = type_full_name
        self.literal = literal

    def __eq__(self, o):
        if not isinstance(o, DefComponent):
            return False
        else:
            return o.name == self.name and \
                o.type_full_name == self.type_full_name and \
                o.literal == self.literal

    def __str__(self):
        return "({}:{}, is_literal={})".format(self.name, self.type_full_name, self.literal)

    def __hash__(self):
        return hash(self.__dict__)

    @staticmethod
    def from_json(json_dct: Optional[dict]):
        if json_dct is None:
            return None
        else:
            return DefComponent(json_dct['name'], json_dct['typeFullName'])


class ObjectUsageSlice:
    """
    A data slice of an object at the start of its definition until its final usage or the data-flow is killed.
    """

    def __init__(self,
                 target_obj: DefComponent,
                 defined_by: Optional[DefComponent] = None,
                 invoked_calls: Union[List[CallEntry], List[str]] = None,
                 arg_to_calls: Union[List[Tuple[CallEntry, int]], List[Tuple[str, int]]] = None,
                 ):
        """
        Instantiates an object slice. Note that names of APIs under invoked_calls and arg_to_calls can be strings.

        :param target_obj: the name and type of the focus object.
        :param defined_by: the name of the call, identifier, or literal that defined the target object, if available.
        :param invoked_calls: calls this object is observed to call.
        :param arg_to_calls: observed calls that this receiver is given to an argument to or returned from. This is
         given as a list of tuples where the first argument is an APIEntry of the procedure call and the second argument
         is where the receiver goes, i.e., (APIEntry, -1) receiver is returned from here or (APIEntry, x > 1) the
         parameter number.
        """
        self.target_obj = target_obj
        self.defined_by = defined_by

        def normalize_call_entry(x):
            if isinstance(x, str):
                return CallEntry(x)
            elif isinstance(x, CallEntry):
                return CallEntry(x.name, x._parameter_entries, x.return_entries)
            else:
                return x

        if invoked_calls is None:
            self.invoked_calls = tuple([])
        else:
            self.invoked_calls = tuple(map(normalize_call_entry, invoked_calls))
        if arg_to_calls is None:
            self.arg_to_calls = tuple([])
        else:
            self.arg_to_calls = tuple(map(lambda x: (normalize_call_entry(x[0]), x[1]), arg_to_calls))

    @staticmethod
    def from_json(json_dct: dict):
        return ObjectUsageSlice(
            DefComponent.from_json(json_dct['targetObj']),
            DefComponent.from_json(json_dct['definedBy']),
            list(map(lambda x: CallEntry.from_json(x), json_dct['invokedCalls'])),
            list(map(lambda x: (CallEntry.from_json(x[0]), x[1]), json_dct['argToCalls'])),
        )

    def __eq__(self, o):
        if not isinstance(o, ObjectUsageSlice):
            return False
        else:
            return o.target_obj == self.target_obj and \
                o.defined_by == self.defined_by and \
                o.invoked_calls == self.invoked_calls and \
                o.arg_to_calls == self.arg_to_calls

    def __str__(self):
        return "Slice(obj: {}, defined_by: {}, #invokes: {}, #used_in: {})".format(self.target_obj.name,
                                                                                   self.defined_by.name,
                                                                                   len(self.invoked_calls),
                                                                                   len(self.arg_to_calls))

    def __repr__(self):
        return str(self.__dict__)

    def __hash__(self):
        return hash(self.__dict__)


class UserDefinedType(TypeEntry):
    """Represents a type defined within the application."""

    def __init__(self, name: str, fields: List[DefComponent], api_entries: List[CallEntry]):
        super().__init__(name, api_entries)
        self.fields = fields

    @staticmethod
    def from_json(json_dct: dict):
        return UserDefinedType(
            json_dct['name'],
            json_dct['fields'],
            json_dct['procedures']
        )

    def __eq__(self, o):
        if not isinstance(o, UserDefinedType):
            return False
        else:
            return super().__eq__(o) and o.fields == self.fields

    def __repr__(self):
        return str(self.__dict__)

    def __hash__(self):
        return hash(self.__dict__)


class MethodUsageSlice:
    """Represents a method's code and the corresponding object usage slices"""

    def __init__(self, source: str, slices: List[ObjectUsageSlice] = None):
        self.source = source
        if slices is None:
            self.slices = tuple([])
        else:
            self.slices = tuple(slices)

    @staticmethod
    def from_json(json_dct: dict):
        def deser_slices(slices: List[Dict]):
            return list(map(lambda x: ObjectUsageSlice.from_json(x), slices))

        return MethodUsageSlice(
            json_dct['source'],
            deser_slices(json_dct['slices'])
        )

    def __repr__(self):
        return str(self.__dict__)

    def __hash__(self):
        return hash(self.__dict__)


class ProgramUsageSlice:
    """Represents all slices and user-defined types within a program."""

    def __init__(self, object_slices: Dict[str, MethodUsageSlice], user_defined_types: List[UserDefinedType]):
        self.object_slices = object_slices
        self.user_defined_types = user_defined_types

    @staticmethod
    def from_json(json_dct: dict):
        return ProgramUsageSlice(
            dict(map(lambda kv: (kv[0], MethodUsageSlice.from_json(kv[1])), dict(json_dct['objectSlices']).items())),
            list(map(lambda x: UserDefinedType.from_json(x), json_dct['userDefinedTypes']))
        )

    def __eq__(self, o):
        if not isinstance(o, ProgramUsageSlice):
            return False
        else:
            return o.object_slices == self.object_slices and \
                o.user_defined_types == self.user_defined_types

    def __repr__(self):
        return str(self.__dict__)

    def __hash__(self):
        return hash(self.__dict__)


class InferenceProblem:
    """
    Provides a specification to define a type inference problem for the TypeInferenceModel to solve.
    """

    def __init__(self,
                 scope_name: str,
                 slices: List[ObjectUsageSlice] = None,
                 ):
        """
        Instantiates an instance of a type inference problem.

        :param scope_name: The name of the scope grouping the object slices together. This is usually the method full
         name.
        :param slices: The observed object slices.
        """
        self.scope_name = scope_name
        if slices is None:
            self.slices = tuple([])
        else:
            self.slices = tuple(slices)


class InferenceSolution:
    """
    Gives a detailed solution to an InferenceProblem.
    """

    def __init__(self,
                 target_identifier: str,
                 inferred_type: str,
                 inferred_type_distance: float,
                 scope: str,
                 ranking: List[WeightedSingleton] = None):
        """
        Instantiates a type inference solution.

        :param target_identifier: The target object's identifier.
        :param inferred_type: The inferred type with the highest ranking.
        :param inferred_type_distance: The distance between the highest ranked type and the next. If there is only one
         suggested type the score will be at the maximum distance of 1.0. Confidence in the inferred type should be
         taken into context with this attribute as if this is too small there may be "no clear winner".
        :param scope: the scope identifier in which this target identifier is valid.
        :param ranking: A list of ranked types with the related score as a weight between [0.0, 1.0].
        """
        self.target_identifier = target_identifier
        self.inferred_type = inferred_type
        self.inferred_type_distance = inferred_type_distance
        self.scope = scope
        if ranking is None:
            self.ranking = tuple([])
        else:
            self.ranking = tuple(ranking)

    def __str__(self):
        return "InferenceSolution(id={}, sol={},ranking={})".format(self.target_identifier, self.inferred_type,
                                                                    self.ranking)

    def __repr__(self):
        return str(self.__dict__)


class BagOfWordsCorpus:
    """
    Represents a corpus of call entries. This class is used for serialization.
    """

    def __init__(self, types: List[TypeEntry]):
        self.types = types

    def to_json(self) -> dict:
        return {
            'types': list(map(lambda x: TypeEntry.to_json(x), self.types))
        }

    @staticmethod
    def from_json(json_dct: dict):
        return BagOfWordsCorpus(
            list(map(lambda x: TypeEntry.from_json(x), json_dct['types'])),
        )


class TypeInferenceModel:
    """
    The basic framework for a type inference model that solves given type inference problems.
    """

    def solve(self, problem: InferenceProblem) -> List[InferenceSolution]:
        """
        Attempts to solve an InferenceProblem.
        :param problem: a specification of observations about receivers and invoked procedures within a scope.
        :return: a solution which describes the most likely type and potential alternatives. If there is no similar type
         then the result will be empty.
        """
        pass

    def usage_slice_to_inference_problems(self, program_slice: ProgramUsageSlice, filtered: Optional[bool] = True) -> \
            List[InferenceProblem]:
        """
        Converts a usage slice extracted from a CPG into an inference problem that can be used by an inference model.
        :param program_slice: a usage slice extracted from a CPG.
        :return: a list of inference problems to classify.
        """

        def to_inference_problem(scope_name: str, slices: List[ObjectUsageSlice], filtered: bool) -> InferenceProblem:
            if filtered:
                slices_to_infer = list(
                    filter(lambda x: x.target_obj.type_full_name == "ANY" or x.target_obj.type_full_name == "UNKNOWN",
                           slices))
            else:
                slices_to_infer = slices
            return InferenceProblem(scope_name, slices_to_infer)

        return list(map(lambda x: to_inference_problem(x[0], x[1], filtered), program_slice.object_slices.items()))


class MalformedAPIEntryException(Exception):
    """
    Raised in the case of an incorrectly defined API entry.
    """

    def __init__(self, given_str: str):
        """
        Instantiates a MalformedAPIEntry exception.

        :param given_str: the API string originally given to parse.
        """
        self.message = "Failed to parse the API definition '{}'! ".format(given_str) + \
                       "Use the format <name>(<param_type1>,<param_type2>,<param_typeN>):<return_type>."
        super().__init__(self.message)


class MalformedTypeEntryException(Exception):
    """
    Raised in the case of an incorrectly defined type entry.
    """

    def __init__(self, given_str: str):
        """
        Instantiates a MalformedTypeEntry exception.

        :param given_str: the type entry string originally given to parse.
        """
        self.message = "Failed to parse type definition '{}'! ".format(given_str) + \
                       " Use the format <type_name>|<api1>;<api2>;<apiN>. " + \
                       " If no APIs are known, simply omit them."
        super().__init__(self.message)
