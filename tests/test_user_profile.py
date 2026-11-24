import unittest

from services.user_profile import needs_user_name, validate_user_name


class UserProfileTests(unittest.TestCase):
    def test_name_prompt_requires_final_identity_and_missing_name(self) -> None:
        self.assertFalse(needs_user_name(False, ""))
        self.assertTrue(needs_user_name(True, ""))
        self.assertFalse(needs_user_name(True, "  Asep  "))

    def test_name_is_trimmed_and_allows_sundanese_letters(self) -> None:
        self.assertEqual(validate_user_name("  Éta Asep-2.  "), "Éta Asep-2.")

    def test_name_rejects_bad_length_url_controls_and_markup(self) -> None:
        for value in ("A", "A" * 31, "https://example.com", "www.example.com", "A\nB", "<b>Asep</b>"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_user_name(value)


if __name__ == "__main__":
    unittest.main()
