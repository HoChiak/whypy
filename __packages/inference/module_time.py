# -*- coding: utf-8 -*-

# import built in libarys

# import 3rd party libarys

# import local libarys
from whypy.__packages.utils import utils


###############################################################################
class SteadyState():
    """
    Class for the steady state case (independence of time).
    """
    attr_time = 'steadystate'

    def __init__(self):
        """
        Class constructor for the steady state case (observations are
        independent of time). Assigns steady state specific methods to the
        inference model.
        """
        self._t = 0

    def adjust4time(self):
        """
        Method to adjust combinations and TBD for time shift. For steadystate
        no action is required.
        """


class Transient():
    """
    Class for the transient case (dependence of time).
    """
    attr_time = 'transient'

    def __init__(self):
        """
        Class constructor for the transient case (observations are
        dependent of time). Assigns transient specific methods to the
        inference model.
        """
