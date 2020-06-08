

class StatusCode(object):
    Success = 0                 # std. result; related to C 'EXIT_SUCCESS'
    # ==== INTERNAL OPERATIONS ==== #
    Repeat = 1
    Delete = 2
    Merge = 3
    Split = 4
    # ==== ERROR CODES ==== #
    Error = 5
    ErrorInterpolation = 51
    ErrorOutOfBounds = 6
    ErrorThroughSurface = 61
    ErrorTimeExtrapolation = 62

