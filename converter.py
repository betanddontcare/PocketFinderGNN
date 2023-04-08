def getFacesRecordsIdx(data):
  counter = True
  faces_records_idx = []
  i = 0

  while counter:
    idx = data.find("FACE" + str(i) + '\'')
    if idx != -1:
      faces_records_idx.append(idx)
      i += 1
    else:
      counter = False
  return faces_records_idx

def getFacesRecords(data):
  faces_records_idx = getFacesRecordsIdx(data)
  faces_len = len(faces_records_idx)
  advanced_faces_records = []
  count = 0
  
  while count < faces_len:
    rest = data[faces_records_idx[count]:]
    cut_record_right = rest.find(')')
    advanced_faces_records.append(data[faces_records_idx[count]:faces_records_idx[count]+cut_record_right])
    count += 1
  return advanced_faces_records

def getFaceBoundsIDs(data):
  advanced_faces_records = getFacesRecords(data)
  ids_from_faces_records = []
  for record in advanced_faces_records:
    cut_record_left = record.find('(')
    ids_from_faces_records.append(record[cut_record_left+1:].split(','))
  return ids_from_faces_records

def getCorrectFaceBoundsIDs(data):
  ids_from_faces_records = getFaceBoundsIDs(data)
  face_bounds_records_ids = []
  for list_of_ids in ids_from_faces_records:
    corrected_face_bound_id = []
    for single_id in list_of_ids:
      if single_id.find('#') == -1:
        single_id = '#' + single_id
        corrected_face_bound_id.append(single_id)
      else:
        corrected_face_bound_id.append(single_id)
    face_bounds_records_ids.append(corrected_face_bound_id)
  return face_bounds_records_ids

def getFaceBoundsRecords(data):
  face_bounds_records_ids = getCorrectFaceBoundsIDs(data)
  bounds_records = []
  for bounds_ids in face_bounds_records_ids:
    bound_id = []
    for id in bounds_ids:
      cut_record_left = data.find(id+'=')
      rest = data[cut_record_left:]
      cut_record_right = rest.find(')')
      bound_id.append(data[cut_record_left:cut_record_left+cut_record_right])
    bounds_records.append(bound_id)
  return bounds_records

def get_edge_loops_ids(data):
  bounds_records = getFaceBoundsRecords(data)
  edge_loop_records_ids = []
  for face_bound in bounds_records:
    face_bound_id = []
    for bound in face_bound:
      cut_record_left = bound.find(',')
      rest = bound[cut_record_left+1:]
      cut_record_right = rest.find(',')
      face_bound_id.append(rest[:cut_record_right])
    edge_loop_records_ids.append(face_bound_id)
  return edge_loop_records_ids

def getEdgeLoopRecords(data):
  edge_loop_records_ids = get_edge_loops_ids(data)
  edge_loop_records = []
  for face_loops in edge_loop_records_ids:
    loop_in_edge_loops = []
    for loop in face_loops:
      cut_record_left = data.find(loop+'=')
      rest = data[cut_record_left:]
      cut_record_right = rest.find(')')
      loop_in_edge_loops.append(data[cut_record_left:cut_record_left+cut_record_right])
    edge_loop_records.append(loop_in_edge_loops)
  return edge_loop_records

def get_oriented_edges_ids(data):
  edge_loop_records = getEdgeLoopRecords(data) 
  sticked_oriented_edges_ids = []
  for loop in edge_loop_records:
    oriented_edge_ids = []
    for element in loop:
      cut_record_left = element.find(',')
      rest = element[cut_record_left+2:]
      oriented_edge_ids.append(rest)
    sticked_oriented_edges_ids.append(oriented_edge_ids)
  return sticked_oriented_edges_ids

def separate_oriented_edges_ids(data):
  sticked_oriented_edges_ids = get_oriented_edges_ids(data)
  oriented_edge_records_ids = []
  for sticked_sample in sticked_oriented_edges_ids:
    separated_ids = []
    for string_element in sticked_sample:
      if string_element.find(',') > 1:
        splited_ids = string_element.split(',')
        for single_id in splited_ids:
          separated_ids.append(single_id)
      else:
        separated_ids.append(string_element)
    oriented_edge_records_ids.append(separated_ids)
  return oriented_edge_records_ids

def getOrientedEdgeRecords(data):
  oriented_edge_records_ids = separate_oriented_edges_ids(data)
  oriented_edge_records = []
  for record_ids in oriented_edge_records_ids:
    edge_with_id = []
    for id in record_ids:
      cut_record_left = data.find(id+'=')
      rest = data[cut_record_left:]
      cut_record_right = rest.find(')')
      edge_with_id.append(data[cut_record_left:cut_record_left+cut_record_right])
    oriented_edge_records.append(edge_with_id)
  return oriented_edge_records

def getIDsOfEdgeCurveRecords(data):
  oriented_edges_records = getOrientedEdgeRecords(data)
  edge_curve_records_ids = []
  for records_for_face in oriented_edges_records:
    edge_curve_ids = []
    for record in records_for_face:
      cut_record_left = record.find('*,*,')
      rest = record[cut_record_left+4:]
      cut_record_right = rest.find(',')
      edge_curve_ids.append(rest[:cut_record_right])
    edge_curve_records_ids.append(edge_curve_ids)
  return edge_curve_records_ids

def getEdgeCurveRecords(data):
  edge_curve_records_ids = getIDsOfEdgeCurveRecords(data)
  edges_curve_records = []
  for record_ids in edge_curve_records_ids:
    edge_with_id = []
    for id in record_ids:
      cut_record_left = data.find(id+'=')
      rest = data[cut_record_left:]
      cut_record_right = rest.find(')')
      edge_with_id.append(data[cut_record_left:cut_record_left+cut_record_right])
    edges_curve_records.append(edge_with_id)
  return edges_curve_records

def extractEdgesWithIDs(data):
  records_with_edge_curve = getEdgeCurveRecords(data)
  edges_assigned_to_faces = []
  for records_for_face in records_with_edge_curve:
    single_edge = []
    for record in records_for_face:
      cut_record_left = record.find('(')
      rest = record[cut_record_left+2:]
      cut_record_right = rest.find(',')
      single_edge.append(rest[:cut_record_right-1])
    edges_assigned_to_faces.append(single_edge)
  return edges_assigned_to_faces

def getFacesAssignedToEdges(data):
  edges_assigned_to_faces = extractEdgesWithIDs(data)
  list_of_faces = []
  i = 0
  while i < data.count('EDGE_CURVE'):
    for edges_list in edges_assigned_to_faces:
      for edge in edges_list:
        if edge == ('EDGE'+ str(i)):
          list_of_faces.append('FACE' + str(edges_assigned_to_faces.index(edges_list)))
    i += 1
  return list_of_faces

def assignEdgesToFaces(data):
  list_of_faces = getFacesAssignedToEdges(data)
  paired_faces = [list(a) for a in zip(list_of_faces[::2], list_of_faces[1::2])]
  count = 0
  for i in paired_faces:
    i.insert(0, 'EDGE' + str(count))
    count += 1
  return paired_faces

def getFacesRecordsToCheckType(data):
  faces_records_idx = getFacesRecordsIdx(data)
  faces_len = len(faces_records_idx)
  advanced_faces_records = []
  count = 0
  
  while count < faces_len:
    rest = data[faces_records_idx[count]:]
    cut_record_left = rest.find('),')
    cut_record_right = rest.find(',.')
    advanced_faces_records.append(rest[cut_record_left+2:cut_record_right])
    count += 1
  return advanced_faces_records

def checkFaceType(rest):
  if rest.startswith('SPHERICAL'):
    return 'Spherical'
  elif rest.startswith('PLAN'):
    return 'Planar'
  elif rest.startswith('CYLINDRI'):
    return 'Cylindrical'
  elif rest.startswith('SURFACE_OF_LINEAR'):
    return 'Swept'
  elif rest.startswith('TOROIDAL'):
    return 'SurfaceOfRevolution'
  elif rest.startswith('OFFSET'):
    return 'Offset'
  elif rest.startswith('CONICAL'):
    return 'Conical'
  elif rest.startswith('B_SPLINE'):
    return 'Parametric'
  else:
    return 'Unknown'

def getFacesTypes(data):
  faces_types_ids = getFacesRecordsToCheckType(data)
  faces_len = len(faces_types_ids)
  faces_with_types = []
  i = 0

  for tp in faces_types_ids:
    face_name = 'FACE' + str(i)
    type_record_idx = data.find(tp + '=')
    rest = data[type_record_idx + len(tp+'='):type_record_idx+30]  
    type_of_face = checkFaceType(rest)
    faces_with_types.append([face_name, type_of_face])
    i += 1
  
  return faces_with_types