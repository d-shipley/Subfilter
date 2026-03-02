"""
Created on Mon Aug  2 12:09:37 2021.

@author: Peter Clark
"""
import subfilter
from loguru import logger


def get_default_variable_list() :
    """
    Provide default variable list.

    Returns
    -------
           var_list.

    The default is::

     var_list = [
            "u",
            "v",
            "w",
            "th",
            "th_v",
            "th_L",
            "q_vapour",
            "q_cloud_liquid_mass",
            "q_total"]

    @author: Peter Clark

    """
    logger.info(f"test_level: {subfilter.global_config['test_level']}")
    if subfilter.global_config['test_level'] == 1:
# For testing
        var_list = [
            "u",
            "v",
            "w",
            "th",
            ]
    elif subfilter.global_config['test_level'] == 2:
# For testing
        var_list = [
            "u",
            "w",
            "th_L",
            "q_cloud_liquid_mass",
            "q_total",
            "cloud_fraction"
            ]
    else:
        var_list = [
            "u",
            "v",
            "w",
            "th",
            "th_v",
            "th_L",
            "q_vapour",
            "q_cloud_liquid_mass",
            "q_total",
            "cloud_fraction"
            ]

    return var_list

def get_default_variable_pair_list() :
    """
    Provide default variable pair list.

    Returns
    -------
        list : list of lists of pairs strings representing variable names.

    The default is::

        var_list = [
                ["u","u"],
                ["u","v"],
                ["u","w"],
                ["v","v"],
                ["v","w"],
                ["w","w"],
                ["u","th"],
                ["v","th"],
                ["w","th"],
                ["u","th_v"],
                ["v","th_v"],
                ["w","th_v"],
                ["u","th_L"],
                ["v","th_L"],
                ["w","th_L"],
                ["u","q_vapour"],
                ["v","q_vapour"],
                ["w","q_vapour"],
                ["u","q_cloud_liquid_mass"],
                ["v","q_cloud_liquid_mass"],
                ["w","q_cloud_liquid_mass"],
                ["u","q_total"],
                ["v","q_total"],
                ["w","q_total"],
                ["th_L","th_L"],
                ["th_L","q_total"],
                ["q_total","q_total"],
                ["th_L","q_vapour"],
                ["th_L","q_cloud_liquid_mass"],
              ]

    @author: Peter Clark

    """
    if subfilter.global_config['test_level'] == 1:
# For testing
        var_list = [
                ["w","th"],
              ]
    elif subfilter.global_config['test_level'] == 2:
# For testing
        var_list = [
                ["u","w"],
                ["w","w"],
                ["u","th"],
                ["w","th"],
                ["w","th_L"],
                ["w","q_total"],
                ["th_L","th_L"],
                ["th_L","q_total"],
                ["q_total","q_total"],
              ]
    else:
        var_list = [
                ["u","u"],
                ["u","v"],
                ["u","w"],
                ["v","v"],
                ["v","w"],
                ["w","w"],
                ["u","th"],
                ["v","th"],
                ["w","th"],
                ["u","th_v"],
                ["v","th_v"],
                ["w","th_v"],
                ["u","th_L"],
                ["v","th_L"],
                ["w","th_L"],
                ["u","q_vapour"],
                ["v","q_vapour"],
                ["w","q_vapour"],
                ["u","q_cloud_liquid_mass"],
                ["v","q_cloud_liquid_mass"],
                ["w","q_cloud_liquid_mass"],
                ["u","q_total"],
                ["v","q_total"],
                ["w","q_total"],
                ["th_L","th_L"],
                ["th_L","q_total"],
                ["q_total","q_total"],
                ["th_L","q_vapour"],
                ["th_L","q_cloud_liquid_mass"],
              ]
    return var_list
