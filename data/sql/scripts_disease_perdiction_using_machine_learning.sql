SELECT * FROM disease_prediction_training WHERE fluid_overload = 1;


SELECT name AS column_names
FROM pragma_table_info('disease_prediction_training');

SELECT COUNT(*), label FROM (
SELECT DISTINCT itching,'itching' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT skin_rash,'skin_rash' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT nodal_skin_eruptions,'nodal_skin_eruptions' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT continuous_sneezing,'continuous_sneezing' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT shivering,'shivering' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT chills,'chills' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT joint_pain,'joint_pain' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT stomach_pain,'stomach_pain' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT acidity,'acidity' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT ulcers_on_tongue,'ulcers_on_tongue' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT muscle_wasting,'muscle_wasting' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT vomiting,'vomiting' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT burning_micturition,'burning_micturition' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT spotting_,'spotting_' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT fatigue,'fatigue' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT weight_gain,'weight_gain' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT anxiety,'anxiety' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT cold_hands_and_feets,'cold_hands_and_feets' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT mood_swings,'mood_swings' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT weight_loss,'weight_loss' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT restlessness,'restlessness' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT lethargy,'lethargy' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT patches_in_throat,'patches_in_throat' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT irregular_sugar_level,'irregular_sugar_level' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT cough,'cough' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT high_fever,'high_fever' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT sunken_eyes,'sunken_eyes' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT breathlessness,'breathlessness' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT sweating,'sweating' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT dehydration,'dehydration' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT indigestion,'indigestion' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT headache,'headache' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT yellowish_skin,'yellowish_skin' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT dark_urine,'dark_urine' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT nausea,'nausea' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT loss_of_appetite,'loss_of_appetite' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT pain_behind_the_eyes,'pain_behind_the_eyes' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT back_pain,'back_pain' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT constipation,'constipation' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT abdominal_pain,'abdominal_pain' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT diarrhoea,'diarrhoea' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT mild_fever,'mild_fever' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT yellow_urine,'yellow_urine' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT yellowing_of_eyes,'yellowing_of_eyes' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT acute_liver_failure,'acute_liver_failure' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT fluid_overload_2,'fluid_overload_2' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT swelling_of_stomach,'swelling_of_stomach' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT swelled_lymph_nodes,'swelled_lymph_nodes' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT malaise,'malaise' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT blurred_and_distorted_vision,'blurred_and_distorted_vision' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT phlegm,'phlegm' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT throat_irritation,'throat_irritation' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT redness_of_eyes,'redness_of_eyes' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT sinus_pressure,'sinus_pressure' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT runny_nose,'runny_nose' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT congestion,'congestion' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT chest_pain,'chest_pain' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT weakness_in_limbs,'weakness_in_limbs' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT fast_heart_rate,'fast_heart_rate' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT pain_during_bowel_movements,'pain_during_bowel_movements' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT pain_in_anal_region,'pain_in_anal_region' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT bloody_stool,'bloody_stool' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT irritation_in_anus,'irritation_in_anus' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT neck_pain,'neck_pain' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT dizziness,'dizziness' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT cramps,'cramps' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT bruising,'bruising' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT obesity,'obesity' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT swollen_legs,'swollen_legs' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT swollen_blood_vessels,'swollen_blood_vessels' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT puffy_face_and_eyes,'puffy_face_and_eyes' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT enlarged_thyroid,'enlarged_thyroid' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT brittle_nails,'brittle_nails' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT swollen_extremeties,'swollen_extremeties' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT excessive_hunger,'excessive_hunger' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT extra_marital_contacts,'extra_marital_contacts' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT drying_and_tingling_lips,'drying_and_tingling_lips' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT slurred_speech,'slurred_speech' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT knee_pain,'knee_pain' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT hip_joint_pain,'hip_joint_pain' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT muscle_weakness,'muscle_weakness' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT stiff_neck,'stiff_neck' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT swelling_joints,'swelling_joints' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT movement_stiffness,'movement_stiffness' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT spinning_movements,'spinning_movements' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT loss_of_balance,'loss_of_balance' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT unsteadiness,'unsteadiness' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT weakness_of_one_body_side,'weakness_of_one_body_side' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT loss_of_smell,'loss_of_smell' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT bladder_discomfort,'bladder_discomfort' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT foul_smell_of,'foul_smell_of' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT continuous_feel_of_urine,'continuous_feel_of_urine' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT passage_of_gases,'passage_of_gases' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT internal_itching,'internal_itching' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT toxic_look_typhos,'toxic_look_typhos' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT depression,'depression' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT irritability,'irritability' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT muscle_pain,'muscle_pain' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT altered_sensorium,'altered_sensorium' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT red_spots_over_body,'red_spots_over_body' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT belly_pain,'belly_pain' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT abnormal_menstruation,'abnormal_menstruation' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT dischromic,'dischromic' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT watering_from_eyes,'watering_from_eyes' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT increased_appetite,'increased_appetite' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT polyuria,'polyuria' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT family_history,'family_history' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT mucoid_sputum,'mucoid_sputum' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT rusty_sputum,'rusty_sputum' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT lack_of_concentration,'lack_of_concentration' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT visual_disturbances,'visual_disturbances' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT receiving_blood_transfusion,'receiving_blood_transfusion' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT receiving_unsterile_injections,'receiving_unsterile_injections' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT coma,'coma' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT stomach_bleeding,'stomach_bleeding' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT distention_of_abdomen,'distention_of_abdomen' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT history_of_alcohol_consumption,'history_of_alcohol_consumption' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT fluid_overload,'fluid_overload' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT blood_in_sputum,'blood_in_sputum' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT prominent_veins_on_calf,'prominent_veins_on_calf' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT palpitations,'palpitations' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT painful_walking,'painful_walking' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT pus_filled_pimples,'pus_filled_pimples' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT blackheads,'blackheads' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT scurring,'scurring' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT skin_peeling,'skin_peeling' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT silver_like_dusting,'silver_like_dusting' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT small_dents_in_nails,'small_dents_in_nails' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT inflammatory_nails,'inflammatory_nails' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT blister,'blister' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT red_sore_around_nose,'red_sore_around_nose' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT yellow_crust_ooze,'yellow_crust_ooze' AS 'label' FROM disease_prediction_training UNION ALL
SELECT DISTINCT prognosis,'prognosis' AS 'label' FROM disease_prediction_training)
GROUP BY
label
HAVING COUNT(*) <> 2;

SELECT prognosis, label FROM (
SELECT DISTINCT itching as symptom ,'itching' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT skin_rash as symptom ,'skin_rash' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT nodal_skin_eruptions as symptom ,'nodal_skin_eruptions' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT continuous_sneezing as symptom ,'continuous_sneezing' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT shivering as symptom ,'shivering' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT chills as symptom ,'chills' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT joint_pain as symptom ,'joint_pain' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT stomach_pain as symptom ,'stomach_pain' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT acidity as symptom ,'acidity' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT ulcers_on_tongue as symptom ,'ulcers_on_tongue' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT muscle_wasting as symptom ,'muscle_wasting' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT vomiting as symptom ,'vomiting' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT burning_micturition as symptom ,'burning_micturition' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT spotting_ as symptom ,'spotting_' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT fatigue as symptom ,'fatigue' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT weight_gain as symptom ,'weight_gain' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT anxiety as symptom ,'anxiety' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT cold_hands_and_feets as symptom ,'cold_hands_and_feets' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT mood_swings as symptom ,'mood_swings' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT weight_loss as symptom ,'weight_loss' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT restlessness as symptom ,'restlessness' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT lethargy as symptom ,'lethargy' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT patches_in_throat as symptom ,'patches_in_throat' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT irregular_sugar_level as symptom ,'irregular_sugar_level' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT cough as symptom ,'cough' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT high_fever as symptom ,'high_fever' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT sunken_eyes as symptom ,'sunken_eyes' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT breathlessness as symptom ,'breathlessness' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT sweating as symptom ,'sweating' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT dehydration as symptom ,'dehydration' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT indigestion as symptom ,'indigestion' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT headache as symptom ,'headache' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT yellowish_skin as symptom ,'yellowish_skin' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT dark_urine as symptom ,'dark_urine' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT nausea as symptom ,'nausea' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT loss_of_appetite as symptom ,'loss_of_appetite' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT pain_behind_the_eyes as symptom ,'pain_behind_the_eyes' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT back_pain as symptom ,'back_pain' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT constipation as symptom ,'constipation' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT abdominal_pain as symptom ,'abdominal_pain' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT diarrhoea as symptom ,'diarrhoea' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT mild_fever as symptom ,'mild_fever' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT yellow_urine as symptom ,'yellow_urine' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT yellowing_of_eyes as symptom ,'yellowing_of_eyes' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT acute_liver_failure as symptom ,'acute_liver_failure' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT fluid_overload_2 as symptom ,'fluid_overload_2' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT swelling_of_stomach as symptom ,'swelling_of_stomach' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT swelled_lymph_nodes as symptom ,'swelled_lymph_nodes' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT malaise as symptom ,'malaise' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT blurred_and_distorted_vision as symptom ,'blurred_and_distorted_vision' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT phlegm as symptom ,'phlegm' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT throat_irritation as symptom ,'throat_irritation' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT redness_of_eyes as symptom ,'redness_of_eyes' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT sinus_pressure as symptom ,'sinus_pressure' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT runny_nose as symptom ,'runny_nose' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT congestion as symptom ,'congestion' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT chest_pain as symptom ,'chest_pain' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT weakness_in_limbs as symptom ,'weakness_in_limbs' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT fast_heart_rate as symptom ,'fast_heart_rate' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT pain_during_bowel_movements as symptom ,'pain_during_bowel_movements' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT pain_in_anal_region as symptom ,'pain_in_anal_region' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT bloody_stool as symptom ,'bloody_stool' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT irritation_in_anus as symptom ,'irritation_in_anus' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT neck_pain as symptom ,'neck_pain' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT dizziness as symptom ,'dizziness' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT cramps as symptom ,'cramps' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT bruising as symptom ,'bruising' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT obesity as symptom ,'obesity' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT swollen_legs as symptom ,'swollen_legs' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT swollen_blood_vessels as symptom ,'swollen_blood_vessels' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT puffy_face_and_eyes as symptom ,'puffy_face_and_eyes' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT enlarged_thyroid as symptom ,'enlarged_thyroid' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT brittle_nails as symptom ,'brittle_nails' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT swollen_extremeties as symptom ,'swollen_extremeties' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT excessive_hunger as symptom ,'excessive_hunger' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT extra_marital_contacts as symptom ,'extra_marital_contacts' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT drying_and_tingling_lips as symptom ,'drying_and_tingling_lips' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT slurred_speech as symptom ,'slurred_speech' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT knee_pain as symptom ,'knee_pain' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT hip_joint_pain as symptom ,'hip_joint_pain' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT muscle_weakness as symptom ,'muscle_weakness' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT stiff_neck as symptom ,'stiff_neck' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT swelling_joints as symptom ,'swelling_joints' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT movement_stiffness as symptom ,'movement_stiffness' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT spinning_movements as symptom ,'spinning_movements' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT loss_of_balance as symptom ,'loss_of_balance' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT unsteadiness as symptom ,'unsteadiness' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT weakness_of_one_body_side as symptom ,'weakness_of_one_body_side' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT loss_of_smell as symptom ,'loss_of_smell' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT bladder_discomfort as symptom ,'bladder_discomfort' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT foul_smell_of as symptom ,'foul_smell_of' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT continuous_feel_of_urine as symptom ,'continuous_feel_of_urine' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT passage_of_gases as symptom ,'passage_of_gases' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT internal_itching as symptom ,'internal_itching' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT toxic_look_typhos as symptom ,'toxic_look_typhos' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT depression as symptom ,'depression' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT irritability as symptom ,'irritability' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT muscle_pain as symptom ,'muscle_pain' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT altered_sensorium as symptom ,'altered_sensorium' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT red_spots_over_body as symptom ,'red_spots_over_body' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT belly_pain as symptom ,'belly_pain' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT abnormal_menstruation as symptom ,'abnormal_menstruation' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT dischromic as symptom ,'dischromic' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT watering_from_eyes as symptom ,'watering_from_eyes' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT increased_appetite as symptom ,'increased_appetite' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT polyuria as symptom ,'polyuria' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT family_history as symptom ,'family_history' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT mucoid_sputum as symptom ,'mucoid_sputum' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT rusty_sputum as symptom ,'rusty_sputum' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT lack_of_concentration as symptom ,'lack_of_concentration' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT visual_disturbances as symptom ,'visual_disturbances' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT receiving_blood_transfusion as symptom ,'receiving_blood_transfusion' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT receiving_unsterile_injections as symptom ,'receiving_unsterile_injections' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT coma as symptom ,'coma' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT stomach_bleeding as symptom ,'stomach_bleeding' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT distention_of_abdomen as symptom ,'distention_of_abdomen' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT history_of_alcohol_consumption as symptom ,'history_of_alcohol_consumption' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT fluid_overload as symptom ,'fluid_overload' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT blood_in_sputum as symptom ,'blood_in_sputum' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT prominent_veins_on_calf as symptom ,'prominent_veins_on_calf' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT palpitations as symptom ,'palpitations' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT painful_walking as symptom ,'painful_walking' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT pus_filled_pimples as symptom ,'pus_filled_pimples' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT blackheads as symptom ,'blackheads' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT scurring as symptom ,'scurring' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT skin_peeling as symptom ,'skin_peeling' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT silver_like_dusting as symptom ,'silver_like_dusting' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT small_dents_in_nails as symptom ,'small_dents_in_nails' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT inflammatory_nails as symptom ,'inflammatory_nails' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT blister as symptom ,'blister' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT red_sore_around_nose as symptom ,'red_sore_around_nose' AS 'label' , prognosis FROM disease_prediction_testing UNION ALL
SELECT DISTINCT yellow_crust_ooze as symptom ,'yellow_crust_ooze' AS 'label' , prognosis FROM disease_prediction_testing)
WHERE symptom = 1
ORDER BY prognosis, label;