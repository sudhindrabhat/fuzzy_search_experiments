from tornado.web import HTTPError



class InternalError(HTTPError):
    display_data = {}

    def __init__(self, http_status=500, error_code=1001, display_data='Something went wrong.'):
        super(InternalError, self).__init__(status_code=http_status)
        self.http_status = http_status
        self.error_code = error_code
        if isinstance(display_data, str):
            self.display_data = {'msg': display_data}
        else:
            self.display_data = display_data

    def set_display_data(self, data):
        if isinstance(data, str):
            self.display_data = {'msg': data}
        else:
            self.display_data = data


class InvalidInput(InternalError):
    def __init__(self, display_data='Invalid input'):
        super(InvalidInput, self).__init__(error_code=1003, http_status=200, display_data=display_data)


class SessionExpired(InternalError):
    def __init__(self):
        super(SessionExpired, self).__init__(error_code=1006, http_status=401, display_data='Session expired.')


class ApiAccessDenied(InternalError):
    def __init__(self):
        super(ApiAccessDenied, self).__init__(error_code=1009, http_status=403, display_data='API access denied.')


class ResourceExists(InternalError):
    def __init__(self):
        super(ResourceExists, self).__init__(error_code=1008, display_data='Resource already exists')