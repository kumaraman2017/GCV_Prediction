from unittest.mock import Mock, call, patch

import run_pipeline


def test_run_pipeline_calls_steps_in_order():
    with patch("run_pipeline.run_clean") as mock_clean, \
         patch("run_pipeline.run_train") as mock_train, \
         patch("run_pipeline.run_explain") as mock_explain:
        manager = Mock()
        manager.attach_mock(mock_clean.main, "clean")
        manager.attach_mock(mock_train.main, "train")
        manager.attach_mock(mock_explain.main, "explain")

        run_pipeline.main()

        manager.assert_has_calls([call.clean(), call.train(), call.explain()])
