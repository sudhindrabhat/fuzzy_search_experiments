import json




class JsonView:
    def __init__(self, data=None, error_code=0):
        self.data = data
        self.error_code = error_code

    def set_data(self, data):
        self.data = data
        return self

    def set_error_code(self, error_code):
        self.error_code = error_code
        return self

    def render(self):
        response = {'error_code': self.error_code}
        if self.data is not None:
            response['data'] = self.data

        return json.dumps(response)
