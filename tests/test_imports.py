def test_data_module_importable():
    from steel_flow.data import generate_csv_files, load_data, clean_column_name
    assert callable(generate_csv_files)
    assert callable(load_data)
    assert callable(clean_column_name)


def test_tune_module_importable():
    from steel_flow.tune import run_study
    assert callable(run_study)


def test_train_module_importable():
    from steel_flow.train import train_model, save_training_config
    assert callable(train_model)
    assert callable(save_training_config)


def test_evaluate_module_importable():
    from steel_flow.evaluate import compute_metrics, plot_error_histogram, plot_shap, plot_prediction_error
    assert callable(compute_metrics)
    assert callable(plot_error_histogram)
    assert callable(plot_shap)
    assert callable(plot_prediction_error)


def test_predict_module_importable():
    from steel_flow.predict import predict_on_sheet, process_excel_file
    assert callable(predict_on_sheet)
    assert callable(process_excel_file)


def test_cli_module_importable():
    from steel_flow.cli import tune, train, predict
    assert callable(tune)
    assert callable(train)
    assert callable(predict)
