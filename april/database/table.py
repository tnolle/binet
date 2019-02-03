# Copyright 2018 Timo Nolle
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================

import json

from sqlalchemy import Column
from sqlalchemy import DATETIME
from sqlalchemy import Float
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from sqlalchemy.orm import relationship

from april.fs import DATABASE_FILE
from april.fs import ModelFile

Base = declarative_base()


def get_engine():
    return create_engine(f'sqlite+pysqlite:///{DATABASE_FILE}')


class Model(Base):
    __tablename__ = 'Model'

    # Keys
    id = Column('Id', Integer, primary_key=True)
    training_event_log_id = Column('TrainingEventLogId', None, ForeignKey('EventLog.Id', name='FK_Model_EventLog'))

    # Model fields
    creation_date = Column('CreationDate', DATETIME)
    algorithm = Column('Algorithm', String)
    training_duration = Column('TrainingDuration', Float)
    file_name = Column('FileName', String)
    training_host = Column('TrainingHost', String)
    hyperparameters = Column('Hyperparameters', String)

    # Relationships
    training_event_log = relationship('EventLog')

    @property
    def json(self):
        if self.training_event_log is not None:
            event_log = self.training_event_log.name
        else:
            event_log = self.file_name.split('_')[0]

        return {
            'id': self.id,
            'creationDate': self.creation_date,
            'algorithm': self.algorithm,
            'trainingEventLog': event_log,
            'trainingDuration': self.training_duration,
            'fileName': self.file_name,
            'trainingHost': self.training_host,
            'hyperparameters': json.loads(self.hyperparameters.replace("'", '"')),
            'modelFileExists': ModelFile(self.file_name).path.exists(),
            'cached': ModelFile(self.file_name).result_file.exists()
        }

    @staticmethod
    def get_model_by_id(id):
        session = Session(get_engine())
        models = [r for r in session.query(Model) if r.id == id]
        return models

    @staticmethod
    def get_json_models():
        session = Session(get_engine())
        models = [r for r in session.query(Model)]
        return models


class EventLog(Base):
    __tablename__ = 'EventLog'

    # Keys
    id = Column('Id', Integer, primary_key=True)
    process_map_id = Column('ProcessMapId', None, ForeignKey('ProcessMap.Id', name='FK_EventLog_ProcessMap'))

    # Event Log fields
    creation_date = Column('CreationDate', DATETIME)
    name = Column('Name', String)
    file_name = Column('FileName', String)
    percent_anomalies = Column('PercentAnomalies', Float)
    num_cases = Column('NumCases', Integer)
    num_activities = Column('NumActivities', Integer)
    num_attributes = Column('NumAttributes', Integer)
    num_events = Column('NumEvents', Integer)
    base_name = Column('BaseName', String(32))
    number = Column('Number', Integer)

    # Relationships
    process_map = relationship('ProcessMap')

    @staticmethod
    def get_eventlogs():
        session = Session(get_engine())
        logs = [r for r in session.query(EventLog)]
        return logs

    @staticmethod
    def get_id_by_name(name):
        session = Session(get_engine())
        q = session.query(EventLog).outerjoin(ProcessMap).filter(EventLog.name == name)
        ids = [r.id for r in q]
        eventlog_id = None
        if len(ids) > 0:
            eventlog_id = ids[0]
        return eventlog_id


class ProcessMap(Base):
    __tablename__ = 'ProcessMap'

    id = Column('Id', Integer, primary_key=True)
    creation_date = Column('CreationDate', DATETIME)
    name = Column('Name', String)
    file_name = Column('FileName', String)
    num_nodes = Column('NumNodes', Integer)
    num_edges = Column('NumEdges', Integer)
    num_traces = Column('NumTraces', Integer)
    max_length = Column('MaxLength', Integer)
    density = Column('Density', Integer)
    in_degree = Column('InDegree', Integer)
    out_degree = Column('OutDegree', Integer)


class Evaluation(Base):
    __tablename__ = 'Evaluation'

    # Keys
    id = Column('Id', Integer, primary_key=True)
    model_id = Column('ModelId', None, ForeignKey('Model.Id', name='FK_Model_Evaluation'))

    # File name for readability
    file_name = Column('FileName', String)

    # Evaluation Fields
    axis = Column('Axis', Integer)
    base = Column('Base', String)
    heuristic = Column('Heuristic', String)
    strategy = Column('Strategy', String)
    label = Column('Label', String)
    perspective = Column('Perspective', String)
    attribute_name = Column('AttributeName', String)
    precision = Column('Precision', Float)
    recall = Column('Recall', Float)
    f1 = Column('F1', Float)

    # Relationships
    model = relationship('Model')


Base.metadata.create_all(get_engine())
