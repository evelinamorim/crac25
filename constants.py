

DEPENDENCY_LABELS = [
    # Core arguments
    "nsubj",  # nominal subject
    "obj",  # direct object
    "iobj",  # indirect object
    "csubj",  # clausal subject
    "ccomp",  # clausal complement
    "xcomp",  # open clausal complement

    # Non-core dependents
    "obl",  # oblique nominal
    "vocative",  # vocative
    "expl",  # expletive
    "dislocated",  # dislocated elements

    # Nominal dependents
    "nmod",  # nominal modifier
    "appos",  # appositional modifier
    "nummod",  # numeric modifier
    "acl",  # clausal modifier of noun
    "amod",  # adjectival modifier
    "det",  # determiner
    "clf",  # classifier
    "case",  # case marking

    # Compound
    "compound",  # compound
    "fixed",  # fixed multiword expression
    "flat",  # flat multiword expression
    "conj",  # conjunct
    "cc",  # coordinating conjunction

    # Function words
    "aux",  # auxiliary
    "cop",  # copula
    "mark",  # marker

    # Modifier words
    "advmod",  # adverbial modifier
    "advcl",  # adverbial clause modifier
    "discourse",  # discourse element

    # Other
    "root",  # root
    "punct",  # punctuation
    "dep",  # unspecified dependency

    # Extended relations
    "nsubj:pass",  # passive nominal subject
    "csubj:pass",  # passive clausal subject
    "acl:relcl",  # relative clause modifier
    "aux:pass",  # passive auxiliary
    "amod:att",  # attributive adjectival modifier
    "nmod:poss",  # possessive nominal modifier
    "compound:prt",  # phrasal verb particle

    # Language-specific relations
    "clf:num",  # numeral classifier
    "nmod:tmod",  # temporal modifier
    "obl:tmod",  # temporal oblique modifier
    "obl:agent",  # agent complement
    "flat:name",  # name
    "nmod:npmod",  # noun phrase as adverbial modifier

    # Additional common relations
    "parataxis",  # parataxis
    "orphan",  # orphan
    "goeswith",  # goes with
    "reparandum",  # overridden disfluency
]
