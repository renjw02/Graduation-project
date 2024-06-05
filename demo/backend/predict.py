import time
import os
import torch
import json
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from tokenizers import AddedToken

from Evaluator import AdvisingEvaluator, RestaurantsEvaluator, WikiSQLEvaluator, IMDBEvaluator

def decode_pqls(
    db_path,
    generator_outputs,
    batch_db_ids,
    tokenizer,
    dataset,
    table_file_path = 'dataset/wikisql/test.tables.jsonl'
):
    batch_size = generator_outputs.shape[0]
    num_return_sequences = generator_outputs.shape[1]

    final_sqls = []
    if dataset == 'advising':
        evaluator = AdvisingEvaluator(db_path)
    elif dataset == 'restaurants':
        evaluator = RestaurantsEvaluator(db_path)
    elif dataset == 'wikisql':
        evaluator = WikiSQLEvaluator(db_path, table_file_path)
    elif dataset == 'imdb':
        evaluator = IMDBEvaluator(db_path)
    else:
        raise ValueError("Invalid dataset name")

    
    for batch_id in range(batch_size):
        pred_executable_sql = "sql placeholder"
        db_id = batch_db_ids[batch_id]

        for seq_id in range(num_return_sequences):
            # pred_sequence是结果
            pred_sequence = tokenizer.decode(generator_outputs[batch_id, seq_id, :], skip_special_tokens = True)
            pred_sql = pred_sequence.split("|")[-1].strip()
            pred_sql = pred_sql.replace("='", "= '").replace("!=", " !=").replace(",", " ,")
            pred_executable_sql = pred_sql
            
            try:
                if dataset == 'wikisql':
                    r = evaluator.execution(pred_executable_sql, db_id)
                else:
                    r = evaluator.execution(pred_executable_sql)
                break
            except Exception as e:
                # print(pred_sql)
                # print(e)
                pass
            
        
        final_sqls.append(pred_executable_sql)
    
    return final_sqls

def model_predict(question: str, device: int = 0, dataset: str = 'default'):
    """
    Predict the pql query based on the question.
    Arguments:
        question: str
        device: int
        dataset: str
    Returns:
        pql: str
    """
    save_path = f"./model/{dataset}"
    db_path = f"./dataset/{dataset}/{dataset}.sqlite"
    data_path = f"./dataset/{dataset}/{dataset}_preprocessed.json"

    start_time = time.time()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = device

    # initialize tokenizer
    tokenizer = T5TokenizerFast.from_pretrained(
        save_path,
        add_prefix_space = True
    )
    
    if isinstance(tokenizer, T5TokenizerFast):
        tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <"), AddedToken(" {"), AddedToken(" }")])

    model_class = T5ForConditionalGeneration

    # initialize model
    model = model_class.from_pretrained(save_path)
    if torch.cuda.is_available():
        model = model.cuda()

    # match processed data

    match_data = {}
    matched = False
    with open(data_path, "r", encoding = 'utf-8') as f:
        data = json.load(f)
    for d in data:
        input_sequence = d['input_sequence'].split('|')[0].strip()
        if question.strip() == input_sequence:
            match_data = d
            matched = True
            break

    if not matched:
        if dataset == 'advising':
            match_data = {
                "input_sequence": question.strip() + ' | sql : comment_instructor , course , course_offering , course_prerequisite , course_tags_count , gsi , instructor , jobs , offering_instructor , program , program_course , program_requirement , requirement , semester , student , student_record , ta | cypher : area | area : area.course_id , area.area | comment_instructor : comment_instructor.instructor_id , comment_instructor.student_id , comment_instructor.score , comment_instructor.comment_text | course : course.course_id , course.name , course.department , course.number , course.credits , course.advisory_requirement , course.enforced_requirement , course.description , course.num_semesters , course.num_enrolled , course.has_discussion , course.has_lab , course.has_projects , course.has_exams , course.num_reviews , course.clarity_score , course.easiness_score , course.helpfulness_score | course_offering : course_offering.offering_id , course_offering.course_id , course_offering.semester , course_offering.section_number , course_offering.start_time , course_offering.end_time , course_offering.monday , course_offering.tuesday , course_offering.wednesday , course_offering.thursday , course_offering.friday , course_offering.saturday , course_offering.sunday , course_offering.has_final_project , course_offering.has_final_exam , course_offering.textbook , course_offering.class_address , course_offering.allow_audit | course_prerequisite : course_prerequisite.pre_course_id , course_prerequisite.course_id | course_tags_count : course_tags_count.course_id , course_tags_count.clear_grading , course_tags_count.pop_quiz , course_tags_count.group_projects , course_tags_count.inspirational , course_tags_count.long_lectures , course_tags_count.extra_credit , course_tags_count.few_tests , course_tags_count.good_feedback , course_tags_count.tough_tests , course_tags_count.heavy_papers , course_tags_count.cares_for_students , course_tags_count.heavy_assignments , course_tags_count.respected , course_tags_count.participation , course_tags_count.heavy_reading , course_tags_count.tough_grader , course_tags_count.hilarious , course_tags_count.would_take_again , course_tags_count.good_lecture , course_tags_count.no_skip | gsi : gsi.course_offering_id , gsi.student_id | instructor : instructor.instructor_id , instructor.name , instructor.uniqname | jobs : jobs.job_id , jobs.job_title , jobs.description , jobs.requirement , jobs.city , jobs.state , jobs.country , jobs.zip | offering_instructor : offering_instructor.offering_instructor_id , offering_instructor.offering_id , offering_instructor.instructor_id | program : program.program_id , program.name , program.college , program.introduction | program_course : program_course.program_id , program_course.course_id , program_course.workload , program_course.category | program_requirement : program_requirement.program_id , program_requirement.category , program_requirement.min_credit , program_requirement.additional_req | requirement : requirement.requirement_id , requirement.requirement , requirement.college | semester : semester.semester_id , semester.semester , semester.year | student : student.student_id , student.lastname , student.firstname , student.program_id , student.declare_major , student.total_credit , student.total_gpa , student.entered_as , student.admit_term , student.predicted_graduation_semester , student.degree , student.minor , student.internship | student_record : student_record.student_id , student_record.course_id , student_record.semester , student_record.grade , student_record.how , student_record.transfer_source , student_record.earn_credit , student_record.repeat_term , student_record.test_id | ta : ta.campus_job_id , ta.student_id , ta.location',
                "db_id": "advising"
            }
        elif dataset == 'restaurants':
            match_data = {
                'input_sequence': question.strip() + ' | sql : restaurant , location | cypher : geographic | restaurant : restaurant.id , restaurant.name , restaurant.food_type , restaurant.city_name , restaurant.rating | location : location.restaurant_id , location.house_number , location.street_name , location.city_name | geographic : geographic.city_name , geographic.county , geographic.region',
                'db_id': 'restaurants'
            }
        elif dataset == 'imdb':
            match_data = {
                'input_sequence': question.strip() + ' | sql : actor , cast , classification , company , copyright , genre , keyword , made_by , producer , tags , tv_series , writer , written_by | cypher : directed_by , director , movie | actor : actor.aid , actor.gender , actor.name , actor.nationality , actor.birth_city , actor.birth_year | cast : cast.id , cast.msid , cast.aid , cast.role | classification : classification.id , classification.msid , classification.gid | company : company.id , company.name , company.country_code | copyright : copyright.id , copyright.msid , copyright.cid | directed_by : directed_by.id , directed_by.msid , directed_by.did | director : director.did , director.gender , director.name , director.nationality , director.birth_city , director.birth_year | genre : genre.gid , genre.genre | keyword : keyword.id , keyword.keyword | made_by : made_by.id , made_by.msid , made_by.pid | movie : movie.mid , movie.title , movie.release_year , movie.title_aka , movie.budget | producer : producer.pid , producer.gender , producer.name , producer.nationality , producer.birth_city , producer.birth_year | tags : tags.id , tags.msid , tags.kid | tv_series : tv_series.sid , tv_series.title , tv_series.release_year , tv_series.num_of_seasons , tv_series.num_of_episodes , tv_series.title_aka , tv_series.budget | writer : writer.wid , writer.gender , writer.name , writer.nationality , writer.birth_city , writer.birth_year | written_by : written_by.id , written_by.msid , written_by.wid',
                'db_id': 'imdb'
            }
        else:
            return None, None, None


    model.eval()
    predict_pql = []
    pql_id = ''
    # print("Matched data:", match_data)

    model_inputs = match_data['input_sequence']
    model_db_ids = match_data['db_id']

    tokenized_inputs = tokenizer(
        model_inputs, 
        return_tensors="pt",
        padding = "max_length",
        max_length = 512,
        truncation = True
    )
    
    encoder_input_ids = tokenized_inputs["input_ids"]
    encoder_input_attention_mask = tokenized_inputs["attention_mask"]
    if torch.cuda.is_available():
        encoder_input_ids = encoder_input_ids.cuda()
        encoder_input_attention_mask = encoder_input_attention_mask.cuda()

    with torch.no_grad():
        model_outputs = model.generate(
            input_ids = encoder_input_ids,
            attention_mask = encoder_input_attention_mask,
            max_length = 512,
            decoder_start_token_id = model.config.decoder_start_token_id,
            num_beams = 8,
            num_return_sequences = 8
        )

        model_outputs = model_outputs.view(1, 8, model_outputs.shape[1])
        predict_pql += decode_pqls(
            db_path, 
            model_outputs, 
            model_db_ids, 
            tokenizer,
            dataset,
        )

        pql_id += model_db_ids
    
    end_time = time.time()
    cost_time = end_time - start_time
    print("Text-to-SQL inference spends {}s.".format(cost_time))

    return predict_pql[0], pql_id, cost_time

def execute(pql: str, table_id: str, dataset: str = 'default', question=None, table_file_path = 'dataset/wikisql/test.tables.jsonl'):
    """
    Execute the pql query and return the results.
    Arguments:
        pql: str
    Returns:
        results: list
    """
    # print('pql:', pql)
    # print('table_id:', table_id)
    # print('dataset:', dataset)

    db_path = f"./dataset/{dataset}/{dataset}.sqlite"
    if dataset == 'advising':
        evaluator = AdvisingEvaluator(db_path)
    elif dataset == 'restaurants':
        evaluator = RestaurantsEvaluator(db_path)
    elif dataset == 'wikisql':
        evaluator = WikiSQLEvaluator(db_path, table_file_path)
    elif dataset == 'imdb':
        evaluator = IMDBEvaluator(db_path)
    else:
        raise ValueError("Invalid dataset name")
    
    results = []
    cols = []
    # execute the query
    try:
        if dataset == 'wikisql':
            results, cols = evaluator.execution(pql, table_id)
        elif dataset == 'imdb':
            results, cols = evaluator._execution(pql, question)
        else:
            results, cols = evaluator.execution(pql)
            # print('cols:', cols)
            # print('results:', results)
        
        if cols is not None:
            cols = add_scope(pql, cols, dataset)
    except Exception as e:
        print(e)
        results, cols = None, None
    return results, cols

def add_scope(pql, cols, dataset):
    for i in range(len(cols)):
        if 'alias' in cols[i]:
            cols[i] = cols[i].replace('alias0', '')
            cols[i] = cols[i].replace('alias1', '')

    if dataset == 'wikisql':
        if 'select' in pql:
            return [f"{col} (relational)" for col in cols]
        else:
            return [f"{col} (graph)" for col in cols]
    elif dataset == 'advising':
        return [f"{col} (relational)" for col in cols]
    elif dataset == 'restaurants':
        return [f"{col} (relational)" for col in cols]
    elif dataset == 'imdb':
        graph = ['movie.release_year', 'movie.title', 'director.birth_city', 'director.birth_year', 'director.gender', 'director.name', 'director.nationality', 'movie.budget']
        columns = ['release_year', 'title', 'birth_city', 'birth_year', 'gender', 'name', 'nationality', 'budget']
        scope = []

        # 检查查询语句的查询列，
        for col in cols:
            if col in columns:
                if graph[columns.index(col)] in pql:
                    scope.append('graph')
                    continue
            scope.append('relational')

        return [f"{col} ({s})" for col, s in zip(cols, scope)]
    
