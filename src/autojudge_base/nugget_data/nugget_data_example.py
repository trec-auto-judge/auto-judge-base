from .nugget_data import (
    Answer,
    NuggetBank,
    NuggetQuestion,
    NuggetClaim,
    Offsets,
    AggregatorType,
    Reference,
    load_nugget_bank_json,
    print_nugget_json,
    question_nugget_to_dict,
    write_nugget_bank_json,
)


def dummy_model() -> NuggetBank:
    nuggets = [
        NuggetQuestion(
            question="What modern conservation efforts are in place to preserve Machu Picchu?",
            question_id="1",
            query_id="1",
            sub_nuggets=[
                NuggetQuestion(
                    query_id="1",
                    question="How are visitor impacts limited?",
                    # , question_id="1.1"
                ).add_answers(
                    [Answer(answer="restricted numbers"), Answer(answer="daily quotas")]
                ),
                NuggetQuestion(
                    query_id="1",
                    question="Which organization sets preservation guidelines?",
                    # , question_id="1.2"
                ).add_answers([Answer(answer="UNESCO")]),
            ],
        ).add_answers(
            [
                Answer(answer="visitor numbers are restricted to reduce wear"),
                Answer(answer="UNESCO provides guidelines for preservation"),
                Answer(
                    answer="restoration projects use traditional materials and methods"
                ),
            ]
        ),
        NuggetQuestion(
            question="How did the Incas likely transport the massive stones used at Machu Picchu?",
            question_id="2",
            query_id="1",
            sub_nuggets=[
                NuggetQuestion(
                    query_id="1",
                    question="Which cylindrical timbers helped move stones?",
                    #   question_id="2.1",
                ).add_answers([Answer(answer="wooden rollers")]),
                NuggetQuestion(
                    query_id="1",
                    question="Which simple machine was used to pry stones?",
                    #   question_id="2.2",
                ).add_answers([Answer(answer="levers")]),
            ],
            aggregator_type=AggregatorType.OR,
            importance=1,
            metadata={"creator": "Laura", "version": "0.0.1"},
        ),
        # example with answer predicates
        NuggetQuestion(
            question="Where is Machu Picchu located?",
            query_id="1",
            aggregator_type=AggregatorType.AND,
        ).add_answers([Answer(answer="in Peru"), Answer(answer="in South America")]),
        # example with references
        NuggetQuestion(
            query_id="1",
            question="When was Machu Picchu built?",
            aggregator_type=AggregatorType.OR,
        ).add_answers(
            [
                Answer.from_lazy(
                    answer="15th Century",
                    references=[
                        "03b53809-fbee-46aa-b036-8934a2b1f556",
                        "0e589240-39b6-49ca-b82f-84f7c55d21f5",
                        "17770e9d-4865-48e5-9140-ffe85401c9d6",
                    ],
                )
            ]
        ),
        NuggetQuestion(
            query_id="1", question="Is machu pichu the right name?"
        ).add_answers([Answer(answer="yes"), Answer(answer="no")]),
        # Claim
        NuggetClaim(
            query_id="1",
            claim="There are unicorns in the world",
            related_question_text="Where are unicorns born?",
            related_answer=[Answer(answer="where their mom is")],
            references=[
                Reference(
                    doc_id="123",
                    collection="fars",
                    offsets=Offsets(start_offset=0, end_offset=10, encoding="UTF-8"),
                )
            ],
        ),
        # minimal example
        NuggetQuestion.from_lazy("1", "Is machu pichu the right name?", ["yes", "no"]),
    ]

    # backwards compatible to v1 style dictionary.
    print("nuggets as v1 style dictionaries")
    print(
        "\n".join(
            [
                str(question_nugget_to_dict(nugget))
                for nugget in nuggets
                if isinstance(nugget, NuggetQuestion)
            ]
        )
    )

    print("add nuggets")
    nuggetBank = NuggetBank(
        query_id="123", title_query="Laura Pichu"
    )  #  , nugget_bank=nuggets)

    nuggetBank.add_nuggets(nuggets)

    print(nuggetBank)

    nuggetBank.index_nuggets()

    print("roots")
    print("\n".join(nuggetBank.nugget_bank.keys()))

    print("all")

    print("\n".join(nuggetBank.all_nuggets_view.keys()))

    return nuggetBank


def main():
    dummy = dummy_model()
    print_nugget_json(dummy)
    print(dummy.model_dump_json(indent=2, exclude_none=True))

    write_nugget_bank_json(dummy, "nugget_bank_example.json.gz")

    x = load_nugget_bank_json("nugget_bank_example.json.gz")


if __name__ == "__main__":
    main()
