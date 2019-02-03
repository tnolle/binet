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

import os
from datetime import datetime
from multiprocessing.pool import Pool

import networkx as nx
import numpy as np
from sqlalchemy.orm import Session
from tqdm import tqdm

from april.database.table import EventLog
from april.database.table import Model
from april.database.table import ProcessMap
from april.database.table import get_engine
from april.fs import PROCESS_MODEL_DIR
from april.fs import get_event_log_files
from april.fs import get_model_files
from april.processmining import ProcessMap as PM
from april.processmining.log import EventLog as EL


def get_dataset(event_log_file):
    engine = get_engine()
    session = Session(engine)

    event_log = EL.load(event_log_file.name)

    q = session.query(ProcessMap).filter(ProcessMap.name == event_log_file.model)
    q = [r.id for r in q]
    if len(q) == 1:
        model_id = q[0]
    else:
        model_id = None

    return {
        'creation_date': datetime.fromtimestamp(os.path.getmtime(event_log_file.path)),
        'process_map_id': model_id,
        'name': event_log_file.name,
        'file_name': event_log_file.file,
        'percent_anomalies': event_log_file.p,
        'num_cases': len(event_log.cases),
        'num_activities': len(event_log.unique_activities),
        'num_attributes': len(event_log.event_attribute_keys),
        'num_events': int(event_log.num_events),
        'base_name': event_log_file.model,
        'number': event_log_file.id
    }


def import_process_maps():
    """
    Read all process model files from disk and write them to the database.

    :return:
    """

    process_maps = []
    process_models = sorted([p for p in PROCESS_MODEL_DIR.glob('*.plg')])
    for process_model in process_models:
        model = PM.from_plg(str(process_model))

        g = model.graph

        process_maps.append({
            'name': process_model.stem,
            'creation_date': datetime.fromtimestamp(os.path.getmtime(str(process_model))),
            'file_name': process_model.name,
            'num_nodes': g.number_of_nodes(),
            'num_edges': g.number_of_edges(),
            'num_traces': len(model.variants.cases),
            'max_length': int(model.variants.max_case_len),
            'density': nx.density(g),
            'in_degree': np.mean([d[1] for d in g.in_degree()]),
            'out_degree': np.mean([d[1] for d in g.out_degree()])
        })

    engine = get_engine()
    session = Session(bind=engine)
    session.bulk_save_objects(ProcessMap(**map) for map in process_maps)
    session.commit()
    session.close()


def import_eventlogs():
    """
    Read all event logs from disk and write them to the database.

    :return:
    """
    existing = [e.file_name for e in EventLog.get_eventlogs()]
    event_logs = [e for e in get_event_log_files() if e.file not in existing]
    event_logs = sorted(event_logs, key=lambda x: x.name)
    datasets = []
    with Pool() as p:
        for dataset in tqdm(p.imap(get_dataset, event_logs), total=len(event_logs), desc='Import Event Logs'):
            datasets.append(dataset)

    engine = get_engine()
    session = Session(bind=engine)
    session.bulk_save_objects(EventLog(**dataset) for dataset in datasets)
    session.commit()
    session.close()


def import_models():
    """
    Read all model files from disk and write them to the database.

    :return:
    """

    engine = get_engine()
    session = Session(bind=engine)

    existing = [m.file_name for m in session.query(Model)]
    model_files = [m for m in get_model_files() if m.file not in existing]
    model_files = sorted(model_files, key=lambda x: x.name)
    models = []

    for model_file in tqdm(model_files, desc='Import Models'):
        q = session.query(EventLog).outerjoin(ProcessMap).filter(EventLog.name == model_file.event_log_name)
        r = [r.id for r in q]
        event_log_id = r[0] if len(r) > 0 else None

        models.append({
            'creation_date': datetime.fromtimestamp(os.path.getmtime(model_file.path)),
            'algorithm': model_file.ad,
            'file_name': model_file.file,
            'training_event_log_id': event_log_id
        })

    session.bulk_save_objects(
        Model(**model) for model in models
    )
    session.commit()


if __name__ == '__main__':
    # import_process_maps()
    import_eventlogs()
    # import_models()
