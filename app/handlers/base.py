import pprint
import tornado
from tornado.web import RequestHandler

from app.exception.customexceptions import InternalError, ApiAccessDenied, SessionExpired
from app.model.user import UserModel
from app.view.templates.json.base import JsonView



class BaseHandler(RequestHandler):
    def set_default_headers(self):
        self.add_header('Content-Type', 'application/json')

    # Uncaught exceptions will be handled here
    def write_error(self, status_code, **kwargs):
        display_data = {'msg': 'Something went wrong'}
        error_code = 1001
        if 'exc_info' in kwargs:
            exception = kwargs['exc_info'][1]
            if isinstance(exception, InternalError):
                display_data = exception.display_data
                error_code = exception.error_code

        if status_code == (404 | 405):
            error_code = status_code
            display_data = None

        json_data = JsonView(display_data, error_code).render()
        self.finish(json_data)

    def get_current_user(self):
        user_session = self.request.headers.get('X-Cbsession')
        user_model = UserModel()
        user_id = user_model.get_user_id_from_session(user_session)
        return user_id

    def get_current_session(self):
        return self.request.headers.get('X-Cbsession')


class BaseAuthenticatedHandler(BaseHandler):
    # Authenticated request. Has to have a valid session. If not, cry!
    def prepare(self):
        super(BaseAuthenticatedHandler, self).prepare()
        if not self.current_user:
            raise SessionExpired


class InternalAuthenticatedHandler(BaseHandler):
    # Authenticated request. Has to have a valid session. If not, cry!
    def prepare(self):
        super(BaseAuthenticatedHandler, self).prepare()
        if not self.current_user:
            raise SessionExpired

    # def get_current_session(self):
    #     return self.request.headers.get('X-Cbsession')

    def get_internal_session(self):
        return self.request.headers.get('X-Cbinternal')
