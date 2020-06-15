

class StatusCode(object):
    Success = 0                 # std. result; related to C 'EXIT_SUCCESS'
    Evaluate = 1
    # ==== INTERNAL OPERATIONS ==== #
    Repeat = 2
    Delete = 3
    Merge = 4
    Split = 5
    StopExecution = 6
    # ==== ERROR CODES ==== #
    Error = 7
    ErrorInterpolation = 71
    ErrorOutOfBounds = 8
    ErrorThroughSurface = 81
    ErrorTimeExtrapolation = 82

