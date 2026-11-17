import warnings

import numpy as np
import xarray as xr

import subfilter.subfilter as sf


def make_dataset(values, name):
    return xr.DataArray(
        values,
        dims=("x_p",),
        coords={"x_p": np.arange(len(values))},
        name=name,
    )


def store_field(dataset, field):
    dataset["ds"][field.name] = field
    return dataset["ds"][field.name]


def test_weighted_filter_field_masks_small_weights_without_division_warnings(
    monkeypatch,
):
    variable = make_dataset([10.0, 20.0, 30.0], "temperature").chunk()
    weight = make_dataset([1.0, 0.0, 1.0], "condition").chunk()
    weight_r = make_dataset([2.0, 0.0, 1e-8], "f(condition)_r").chunk()
    weight_s = make_dataset([0.0, 1.0, 0.0], "f(condition)_s").chunk()
    numerator_r = make_dataset([4.0, 8.0, 12.0], "numerator_r").chunk()
    numerator_s = make_dataset([6.0, 9.0, 15.0], "numerator_s").chunk()

    def filter_field(field, options, filter_def):
        if field.name == weight.name:
            return weight_r, weight_s
        return numerator_r, numerator_s

    monkeypatch.setattr(sf, "filtered_field_calc", filter_field)
    monkeypatch.setattr(sf, "save_field", store_field)
    filtered_dataset = {"ds": xr.Dataset()}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        resolved, subfilter = sf.weighted_filter_field(
            variable,
            filtered_dataset,
            {},
            object(),
            weight=weight,
        )
        resolved = resolved.compute()
        subfilter = subfilter.compute()

    np.testing.assert_allclose(resolved, [2.0, 0.0, 0.0])
    np.testing.assert_allclose(subfilter, [3.0, 0.0, 0.0])
    assert not [
        warning
        for warning in caught
        if issubclass(warning.category, RuntimeWarning)
    ]


def test_weighted_filter_field_without_weight_saves_unweighted_fields(
    monkeypatch,
):
    variable = make_dataset([1.0, 2.0, 3.0], "temperature")
    resolved_field = make_dataset([2.0, 4.0, 6.0], "f(temperature)_r")
    subfilter_field = make_dataset([-1.0, -2.0, -3.0], "f(temperature)_s")
    calls = []

    def filter_field(field, options, filter_def):
        calls.append(field.name)
        return resolved_field, subfilter_field

    monkeypatch.setattr(sf, "filtered_field_calc", filter_field)
    monkeypatch.setattr(sf, "save_field", store_field)
    filtered_dataset = {"ds": xr.Dataset()}

    resolved, subfilter = sf.weighted_filter_field(
        variable,
        filtered_dataset,
        {},
        object(),
        weight=None,
    )

    np.testing.assert_allclose(resolved, resolved_field)
    np.testing.assert_allclose(subfilter, subfilter_field)
    assert calls == ["temperature"]
    assert set(filtered_dataset["ds"].data_vars) == {
        "f(temperature)_r",
        "f(temperature)_s",
    }


def test_weighted_quadratic_subfilter_returns_conditional_pair_flux(
    monkeypatch,
):
    variable1 = make_dataset([2.0, 5.0], "u_on_p")
    variable2 = make_dataset([3.0, 7.0], "v_on_p")
    weight = make_dataset([1.0, 0.0], "mask_indicator")
    weight_r = make_dataset([2.0, 0.0], "f(mask_indicator)_r")
    weight_s = make_dataset([0.0, 1.0], "f(mask_indicator)_s")
    filtered_parts = {
        "mask_indicator.u_on_p": (
            make_dataset([4.0, 10.0], "u_numerator_r"),
            make_dataset([0.0, 0.0], "u_numerator_s"),
        ),
        "mask_indicator.v_on_p": (
            make_dataset([6.0, 14.0], "v_numerator_r"),
            make_dataset([0.0, 0.0], "v_numerator_s"),
        ),
        "mask_indicator.u_on_p.v_on_p": (
            make_dataset([20.0, 30.0], "pair_numerator_r"),
            make_dataset([0.0, 0.0], "pair_numerator_s"),
        ),
    }

    def filter_field(field, options, filter_def):
        if field.name == weight.name:
            return weight_r, weight_s
        return filtered_parts[field.name]

    def get_data(source, reference, name, **kwargs):
        return {"u": variable1, "v": variable2}[name]

    monkeypatch.setattr(sf, "filtered_field_calc", filter_field)
    monkeypatch.setattr(sf, "get_data_on_grid", get_data)
    monkeypatch.setattr(sf, "save_field", store_field)
    filtered_dataset = {"ds": xr.Dataset()}
    derived_dataset = {"ds": xr.Dataset()}

    flux, product, product_r, product_s = sf.weighted_quadratic_subfilter(
        xr.Dataset(),
        None,
        derived_dataset,
        filtered_dataset,
        {},
        object(),
        "u",
        "v",
        weight=weight,
    )

    np.testing.assert_allclose(flux, [4.0, 0.0])
    np.testing.assert_allclose(product, variable1 * variable2)
    np.testing.assert_allclose(product_r, [10.0, 0.0])
    np.testing.assert_allclose(product_s, [0.0, 0.0])
    assert flux.name == "s(u,v)_on_p_mask_indicator-weighted"
