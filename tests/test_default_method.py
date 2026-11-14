import unittest

import main
import read_braille


class EntryPointTests(unittest.TestCase):
    def test_entrypoint_is_six_dot_reader(self):
        self.assertIs(main.main, read_braille.main)

    def test_removed_method_flag_is_rejected(self):
        import contextlib
        import io

        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            main.main(["--method", "text"])
        self.assertEqual(raised.exception.code, 2)
