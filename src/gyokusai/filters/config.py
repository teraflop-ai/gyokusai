from pathlib import Path

BAD_WORDS_PATH = Path(__file__).parents[1] / "word_lists" / "badwords_en.txt"
ENGLISH_STOPWORDS = ["the", "be", "to", "of", "and", "that", "have", "with"]
BOILERPLATE_POLICY_NOTICES = [
    "terms of use",
    "privacy policy",
    "cookie policy",
    "uses cookies",
    "privacy overview",
    "use of cookies",
    "use cookies",
    "privacy & cookies policy",
    "privacy and cookies policy",
    "This website uses cookies to improve your experience while you "
    "navigate through the website. Out of these cookies, the cookies "
    "that are categorized as necessary are stored on your browser as they "
    "are essential for the working of basic functionalities of the website. "
    "We also use third-party cookies that help us analyze and understand how "
    "you use this website. These cookies will be stored in your browser only "
    "with your consent. You also have the option to opt-out of these "
    "cookies. But opting out of some of these cookies may have an effect "
    "on your browsing experience.".lower(),
    "Necessary cookies are absolutely essential for the website to "
    "function properly. This category only includes cookies that "
    "ensures basic functionalities and security features of the website. "
    "These cookies do not store any personal information.".lower(),
    "Any cookies that may not be particularly necessary for the website "
    "to function and is used specifically to collect user personal data "
    "via analytics, ads, other embedded contents are termed as non-necessary "
    "cookies. It is mandatory to procure user consent prior to running these "
    "cookies on your website.".lower(),
    "This site uses cookies, including for analytics, personalization, and "
    "advertising purposes. For more information or to change your "
    "cookie settings, click here.".lower(),
    "If you continue to browse this site without changing your cookie "
    "settings, you agree to this use. AcceptRead More".lower(),
]
