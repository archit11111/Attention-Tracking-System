from numpy.lib.arraysetops import unique
from .db import db
import datetime
import pytz

class TrackingLogs(db.Document):
    name = db.StringField(required=True)
    created_at = db.StringField(required=True)
    finished_at = db.StringField(default='-')
    time_limit = db.IntField(required=True)
    calibrated_values = db.DictField()
    exam_id = db.StringField(required=True)
    answers = db.DictField()
    marks_obtained = db.IntField()
    attention_logs = db.ListField(db.DictField())

# model present in express server
class Exams(db.Document):
    creatorName = db.StringField()
    creatorId = db.StringField()
    examName = db.StringField()
    examDescription = db.StringField()
    examStartTime = db.StringField()
    examEndTime = db.StringField()
    timelimit = db.IntField()
    questions = db.ListField()
    totalMarks = db.IntField()
    v = db.IntField(db_field='__v')
