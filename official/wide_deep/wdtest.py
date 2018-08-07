import tensorflow as tf

feature_names = [
    'age','education']

d = dict(zip(feature_names, [[34], ['Bachelors']])), '>50K'

print(d[0])
age = tf.feature_column.numeric_column('age')
age_buckets = tf.feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
education_vocabulary_list = [
    'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
    'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
    '5th-6th', '10th', '1st-4th', 'Preschool', '12th']
education = tf.feature_column.categorical_column_with_vocabulary_list('education', vocabulary_list=education_vocabulary_list)
eductation_indicator = tf.feature_column.indicator_column(education)
feature_columns = [age_buckets, eductation_indicator]
print(feature_columns)

input_layer = tf.feature_column.input_layer(
    features=d[0],
    feature_columns=feature_columns
)

zero = tf.constant(0, dtype=tf.float32)
where = tf.not_equal(input_layer, zero)
indices = tf.where(where)
values = tf.gather_nd(input_layer, indices)
sparse = tf.SparseTensor(indices, values, input_layer.shape)


with tf.train.MonitoredTrainingSession() as sess:

    print(input_layer)
    print(sess.run(input_layer))
    print(sess.run(sparse))