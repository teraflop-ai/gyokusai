EXTRACT_INSTRUCTION = """Extract the substantive mathematical page content from the given raw web-page plaintext by removing generic website boilerplate and interface text.

The input has already been selected for containing mathematical content. It may be truncated, noisy, malformed, or non-English. Be conservative: when unsure whether text is substantive page content or boilerplate, keep it.

KEEP:
- Page/article/problem titles and substantive headings.
- Mathematical exposition and surrounding explanatory prose.
- Definitions, propositions, lemmas, theorems, corollaries, proofs, derivations, and remarks.
- Exercises, problems, questions, answers, hints, solutions, and worked examples.
- Equations, formulas, symbols, LaTeX, mathematical notation, and displayed mathematics.
- Lists, tables, code, pseudocode, figure/table captions, footnotes, citations, and references when they are part of the substantive content.
- Substantive forum, Q&A, and discussion text, including questions, answers, and comments.
- Local structural labels needed to understand the content, such as "Question", "Answer", "Proof", "Solution 2", exercise numbers, and section numbers.
- Ordinary prose that introduces, explains, motivates, or connects the mathematics, even if that prose itself contains little notation.

REMOVE:
- Site-wide navigation menus, breadcrumbs, category menus, headers, footers, and sidebars.
- Search controls, login/signup/account controls, subscription prompts, cookie/privacy banners, and consent text.
- Advertisements, sponsorship blocks, donation prompts, and promotional material unrelated to the page content.
- "Related", "recommended", "popular", "trending", or other automatically generated link lists.
- Social/share/follow buttons and interface labels.
- Tags, pagination controls, and generic previous/next navigation.
- Generic copyright, terms, privacy, accessibility, and other site-wide legal/footer text.
- Forum/account metadata such as usernames, avatars, timestamps, vote counts, view counts, reputation scores, badges, and edit/flag/report/share controls.
- Repeated template text or other site chrome that would appear essentially unchanged on many pages of the same website.

STRICT EXTRACTION RULES:
- This is deletion-only extraction, not summarization.
- Preserve retained text verbatim and in its original order.
- Preserve the original language, spelling, capitalization, punctuation, notation, LaTeX, code, and line structure of retained content.
- Do not paraphrase, summarize, translate, normalize, correct, rewrite, reorder, infer, complete, or add text.
- Do not remove substantive content merely because it is short, repetitive, informal, or contains little mathematics.
- Do not try to repair truncated or malformed input. If substantive content is truncated, preserve it as truncated.
- If nothing substantive remains after removing boilerplate, output nothing.

Output only the extracted plaintext, with no preamble, explanation, commentary, quotation marks, or code fences."""

GRADE_INSTRUCTION = """Classify the mathematical level and topics of the given text.

Level = minimum background a reader needs to follow the text's main mathematical content. Judge the dominant content, not isolated hard fragments or stray advanced vocabulary. Ignore boilerplate, navigation, ads, and formatting noise. Text may be truncated or non-English; classify what is present.

Levels:
0 = No substantive math (non-math prose, site chrome, education policy/opinion without actual math).
1 = Elementary (K-5): counting, whole-number arithmetic, place value, basic fractions/decimals, time/money, basic shapes and measurement, simple word problems.
2 = Middle school (6-8): ratios/proportions, percents, negative numbers, exponents/roots, order of operations, pre-algebra, one-variable linear equations/inequalities, coordinate plane, area/volume, Pythagorean theorem, mean/median, simple probability.
3 = High school (9-12): Algebra I/II (systems, quadratics, polynomials, rational/radical expressions, exponentials/logarithms), functions and graphs, geometry with proofs, trigonometry, precalculus (complex numbers, vectors, conics, sequences/series), intro statistics, SAT/ACT math, AMC 10/12-level contest problems.
4 = Advanced high school / early undergraduate: single-variable calculus (AP/IB), AP statistics, intro linear algebra, intro proof-writing and discrete math, AIME/USAMO/IMO-level olympiad problems.
5 = Undergraduate: multivariable/vector calculus, ODEs, linear algebra (vector spaces, eigenvalues), real and complex analysis, abstract algebra, elementary number theory, point-set topology, probability theory, mathematical statistics, numerical analysis, graph theory, Putnam-level problems.
6 = Graduate: measure theory, functional analysis, algebraic/differential geometry, algebraic topology, PDE theory, representation theory, commutative algebra, analytic number theory, stochastic calculus, category theory, qualifying-exam level material.
7 = Research: papers, preprints, novel theorems/proofs, open problems, frontier-level discussion. Requires research-level depth or new results, not just graduate vocabulary.

Output exactly one line of JSON, nothing else:
{"level": <0-7>}"""

TOPIC_INSTRUCTION = """Classify the mathematical topics of the given text.

Topics = the main mathematical subject areas of the text's substantive content. Judge the dominant content, not isolated fragments or stray vocabulary. Ignore boilerplate, navigation, ads, and formatting noise. Text may be truncated or non-English; classify what is present.

Topics:
arithmetic = counting, whole numbers, fractions, decimals, percents, ratios/proportions, place value, order of operations, units/measurement, time/money.
algebra = expressions, equations, inequalities, polynomials, functions and graphs, exponentials/logarithms, sequences/series, elementary complex numbers, precalculus.
geometry = plane/solid Euclidean geometry, coordinate geometry, transformations, trigonometry, vectors, conics.
number_theory = divisibility, primes, congruences, Diophantine equations, arithmetic functions, algebraic/analytic number theory.
combinatorics = counting, permutations/combinations, pigeonhole, generating functions, graph theory, discrete structures, enumerative/extremal combinatorics.
probability = probability, random variables, distributions, expectation, Markov chains, stochastic processes.
statistics = descriptive statistics, data displays, sampling, estimation, hypothesis testing, regression, statistical methods.
calculus = limits, derivatives, integrals, infinite series, multivariable/vector calculus.
differential_equations = ODEs, PDEs, dynamical systems, chaos.
analysis = real/complex analysis, measure theory, functional analysis, harmonic analysis, special functions.
linear_algebra = matrices, determinants, vector spaces, linear maps, eigenvalues, inner products, matrix analysis.
abstract_algebra = groups, rings, fields, modules, Galois theory, commutative algebra, representation theory, homological algebra, category theory.
topology_geometry = point-set/algebraic topology, differential geometry, manifolds, Lie groups, algebraic geometry.
logic_foundations = propositional/predicate logic, set theory, proof techniques, model theory, computability, foundations.
optimization = linear/nonlinear programming, operations research, game theory, control, calculus of variations.
computation = numerical analysis, algorithms, complexity, scientific computing, computer algebra, math software.
applied = mathematical physics, mathematical modeling, mathematical finance, cryptography, biology/engineering/economics applications.
other = math history, pedagogy, philosophy of math, recreational puzzles, or math content not fitting above.

Rules:
- Output 1 to 3 topics, ordered by prominence.
- Choose the most specific topic that fits; use "other" only when nothing else applies.
- Output an empty list if the text has no substantive math.

Output exactly one line of JSON, nothing else:
{"topics": ["<topic>", ...]}"""