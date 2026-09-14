import daft

from gyokusai.filters import (
    AlphabeticWordsFilter,
    BadWordsFilter,
    BoilerPlateLineFilter,
    BoilerPlateStringFilter,
    BulletsFilter,
    CurlyBraceFilter,
    DigitRatioFilter,
    EllipsisFilter,
    GzipCompressionFilter,
    JavascriptLineFilter,
    LengthFilter,
    LoremIpsumFilter,
    MeanWordLengthFilter,
    MinLinesFilter,
    MinWordsLineFilter,
    NonAlphaNumericFilter,
    PunctuationFilter,
    RepeatedLinesByCharFilter,
    RepeatedLinesFilter,
    RepeatedParagraphsByCharFilter,
    RepeatedParagraphsFilter,
    RepeatingDuplicateNGramsFilter,
    RepeatingTopNGramsFilter,
    StopWordDensityFilter,
    StopWordFilter,
    SymbolsToWordsFilter,
    TableRatioFilter,
    TerminalPunctuationLineFilter,
    ThreeSentenceDedupFilter,
    WordCountFilter,
)


def test_lorem_ipsum_filter():
    texts = [
        "The library is open every day.",
        "Lorem ipsum dolor sit amet.",
        "This contains LOREM IPSUM placeholder text.",
    ]
    df = daft.from_pydict({"text": texts})

    result = LoremIpsumFilter()(df).to_pydict()

    assert result["text"] == [texts[0]]


def test_curly_brace_filter():
    texts = [
        "The library is open every day.",
        "function example() { return 1; }",
    ]
    df = daft.from_pydict({"text": texts})

    result = CurlyBraceFilter()(df).to_pydict()

    assert result["text"] == [texts[0]]


def test_min_lines_filter():
    texts = [
        "The library is open.\nYou can borrow books.\nEveryone is welcome.",
        "The library is open.\nYou can borrow books.",
    ]
    df = daft.from_pydict({"text": texts})

    result = MinLinesFilter(min_lines=3)(df).to_pydict()

    assert result["text"] == [texts[0]]


def test_terminal_punctuation_line_filter():
    text = "The library is open.\nThis line lacks punctuation\nEveryone is welcome!"
    df = daft.from_pydict({"text": [text]})

    result = TerminalPunctuationLineFilter()(df).to_pydict()

    assert result["text"] == ["The library is open.\nEveryone is welcome!"]


def test_min_words_line_filter():
    text = (
        "These are exactly five words.\n"
        "Only three words.\n"
        "This line contains six whole words."
    )
    df = daft.from_pydict({"text": [text]})

    result = MinWordsLineFilter(min_words=5)(df).to_pydict()

    assert result["text"] == [
        "These are exactly five words.\nThis line contains six whole words."
    ]


def test_javascript_line_filter():
    text = (
        "The library is open.\n"
        "Please enable JavaScript.\n"
        "JAVASCRIPT is required.\n"
        "Everyone is welcome."
    )
    df = daft.from_pydict({"text": [text]})

    result = JavascriptLineFilter()(df).to_pydict()

    assert result["text"] == ["The library is open.\nEveryone is welcome."]


def test_length_filter():
    texts = ["ab", "abc", "abcd", "abcde", "abcdef"]
    df = daft.from_pydict({"text": texts})

    result = LengthFilter(min_len=2, max_len=6)(df).to_pydict()

    assert result["text"] == ["abc", "abcd", "abcde"]


def test_punctuation_filter():
    texts = [
        "The library is open.\nEveryone is welcome",
        "The library is open\nEveryone is welcome",
    ]
    df = daft.from_pydict({"text": texts})

    result = PunctuationFilter(max_ratio=0.5)(df).to_pydict()

    assert result["text"] == [texts[0]]


def test_bad_words_filter():
    texts = [
        "The library is open every day.",
        "This contains a BADWORD.",
        "The token badwordish is allowed.",
    ]
    df = daft.from_pydict({"text": texts})

    result = BadWordsFilter(badwords=["badword"])(df).to_pydict()

    assert result["text"] == [texts[0], texts[2]]


def test_gzip_compression_filter():
    texts = [
        ("The library is open every day. You can borrow books and read newspapers. ")
        * 3,
        "a" * 10000,
        "abc",
    ]
    df = daft.from_pydict({"text": texts})

    result = GzipCompressionFilter()(df).to_pydict()

    assert result["text"] == [texts[0]]


def test_digit_ratio_filter():
    texts = ["abcdefghij", "abcd1", "abcd12"]
    df = daft.from_pydict({"text": texts})

    result = DigitRatioFilter(max_ratio=0.2)(df).to_pydict()

    assert result["text"] == ["abcdefghij", "abcd1"]


def test_non_alpha_numeric_filter():
    texts = ["hello world", "abc@", "abc@@@"]
    df = daft.from_pydict({"text": texts})

    result = NonAlphaNumericFilter(max_ratio=0.25)(df).to_pydict()

    assert result["text"] == ["hello world", "abc@"]


def test_ellipsis_filter():
    texts = [
        "The library is open.\nYou can borrow books...",
        "The library is open...\nYou can borrow books...",
    ]
    df = daft.from_pydict({"text": texts})

    result = EllipsisFilter(max_ratio=0.5)(df).to_pydict()

    assert result["text"] == [texts[0]]


def test_word_count_filter():
    texts = [
        "one two",
        "one two three",
        "one two three four",
        "one two three four five",
    ]
    df = daft.from_pydict({"text": texts})

    result = WordCountFilter(min_words=3, max_words=4)(df).to_pydict()

    assert result["text"] == [texts[1], texts[2]]


def test_repeated_paragraphs_filter():
    texts = [
        ("The library has books.\n\nThe park has trees.\n\nThe museum has paintings."),
        ("The library has books.\n\nThe library has books.\n\nThe library has books."),
    ]
    df = daft.from_pydict({"text": texts})

    result = RepeatedParagraphsFilter(ratio=0.7)(df).to_pydict()

    assert result["text"] == [texts[0]]


def test_three_sentence_dedup_filter():
    texts = ["A.\nB.\nC.", "A.\nB.\nC.", "D.\nE.\nF."]
    df = daft.from_pydict({"record_id": [1, 2, 3], "text": texts})

    result = ThreeSentenceDedupFilter()(df).sort("record_id").to_pydict()

    assert result["record_id"] == [1, 3]
    assert result["text"] == [texts[0], texts[2]]


def test_repeated_lines_filter():
    texts = ["A.\nB.\nC.", "A.\nB.\nC.\nA.", "A.\nB.\nA."]
    df = daft.from_pydict({"text": texts})

    result = RepeatedLinesFilter()(df).to_pydict()

    assert result["text"] == texts[:2]


def test_stop_word_density_filter():
    texts = ["THE cat AND dog", "The cat ran home", "theater candy"]
    df = daft.from_pydict({"text": texts})

    result = StopWordDensityFilter(min_ratio=0.5, stopwords=["the", "and"])(
        df
    ).to_pydict()

    assert result["text"] == [texts[0]]


def test_stop_word_filter():
    texts = ["THE cat AND dog", "The cat ran home", "theater candy", "the the cat"]
    df = daft.from_pydict({"text": texts})

    result = StopWordFilter(min_stop_words=2, stopwords=["the", "and"])(df).to_pydict()

    assert result["text"] == [texts[0], texts[3]]


def test_table_ratio_filter():
    texts = [
        "The library is open.\nEveryone is welcome.",
        "| Name | Age |\nThe library is open.",
        "| Name | Age |\n| Alice | 30 |",
    ]
    df = daft.from_pydict({"text": texts})

    result = TableRatioFilter(max_ratio=0.5)(df).to_pydict()

    assert result["text"] == texts[:2]


def test_alphabetic_words_filter():
    texts = ["one two three", "one two 123 456", "one 123 456"]
    df = daft.from_pydict({"text": texts})

    result = AlphabeticWordsFilter(min_ratio=0.5)(df).to_pydict()

    assert result["text"] == texts[:2]


def test_bullets_filter():
    texts = [
        "The library is open.\nEveryone is welcome.",
        "• First item.\nThis is ordinary prose.",
        "• First item.\n‣ Second item.",
    ]
    df = daft.from_pydict({"text": texts})

    result = BulletsFilter(max_ratio=0.5)(df).to_pydict()

    assert result["text"] == texts[:2]


def test_symbols_to_words_filter():
    texts = [
        "one two three four",
        "# one two three",
        "... one two three",
        "# ... one two",
        "# # one two",
        "... ... one two",
    ]
    df = daft.from_pydict({"text": texts})

    result = SymbolsToWordsFilter(max_ratio=0.25)(df).to_pydict()

    assert result["text"] == texts[:4]


def test_repeating_top_ngrams_filter():
    texts = ["aa bb cc dd", "aa aa aa", "aa bb aa bb", "aa", ""]
    df = daft.from_pydict({"text": texts})

    result = RepeatingTopNGramsFilter(n=2, max_ratio=0.625)(df).to_pydict()

    assert result["text"] == texts[:2]


def test_repeating_duplicate_ngrams_filter():
    texts = ["aa bb cc dd", "aa aa aa aa", "aa aa aa aa aa", "aa", ""]
    df = daft.from_pydict({"text": texts})

    result = RepeatingDuplicateNGramsFilter(n=2, max_ratio=8 / 11)(df).to_pydict()

    assert result["text"] == texts[:2]


def test_repeated_lines_by_char_filter():
    texts = ["A.\nBBB.", "A.\nBBB.\nA.", "A.\nBBB.\nBBB."]
    df = daft.from_pydict({"text": texts})

    result = RepeatedLinesByCharFilter(ratio=0.75)(df).to_pydict()

    assert result["text"] == texts[:2]


def test_repeated_paragraphs_by_char_filter():
    texts = ["A.\n\nBBB.", "A.\n\nBBB.\n\nA.", "A.\n\nBBB.\n\nBBB."]
    df = daft.from_pydict({"text": texts})

    result = RepeatedParagraphsByCharFilter(ratio=0.75)(df).to_pydict()

    assert result["text"] == texts[:2]


def test_mean_word_length_filter():
    texts = [
        "a bb",
        "cat dog",
        "  cat\tfive\n",
        "four five",
        "a!! b??",
        "été déjà",
        "longer words",
        "",
        " \t\n",
        None,
    ]
    df = daft.from_pydict({"text": texts})

    result = MeanWordLengthFilter(min_length=3, max_length=4)(df).to_pydict()

    assert result["text"] == texts[1:6]


def test_boilerplate_string_filter():
    texts = [
        "The library is open.",
        "One.\n\nTwo.\n\nThree.\n\nPRIVACY POLICY.\n\nTerms of use and cookie policy.",
        "Article paragraph.\n\nPrivacy policy.\n\nTerms of use.",
        "The library is open.\n\nLOREM IPSUM placeholder text.",
    ]
    df = daft.from_pydict({"text": texts})

    result = BoilerPlateStringFilter(max_boilerplate_string_ratio=0.4)(df).to_pydict()

    assert result["text"] == texts[:2]


def test_boilerplate_line_filter():
    texts = [
        "PRIVACY POLICY.\nThe library is open.\nThis site uses cookies.",
        "The library is open.\n\nEveryone is welcome.",
        "Privacy policy.\nTerms of use.",
    ]
    df = daft.from_pydict({"text": texts})

    result = BoilerPlateLineFilter()(df).to_pydict()

    assert result["text"] == ["The library is open.", texts[1], ""]
