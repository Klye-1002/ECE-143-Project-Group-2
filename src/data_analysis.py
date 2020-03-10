#library
import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from wordcloud import WordCloud
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import warnings
warnings.filterwarnings("ignore")
pd.set_option('max_colwidth', 500)
pd.set_option('max_columns', 500)
pd.set_option('max_rows', 100)

from sklearn.metrics import cohen_kappa_score
def kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def read_data(filepath):
    breeds = pd.read_csv(filepath + '/breed_labels.csv')
    colors = pd.read_csv(filepath + '/color_labels.csv')
    states = pd.read_csv(filepath + '/state_labels.csv')

    train = pd.read_csv(filepath + '/train/train.csv')
    test = pd.read_csv(filepath + '/test/test.csv')

    train['dataset_type'] = 'train'
    test['dataset_type'] = 'test'
    all_data = pd.concat([train, test])
    all_count = train['AdoptionSpeed'].value_counts(normalize=True).sort_index()

    sentiment_dict = {}
    for filename in os.listdir(filepath + '/train_sentiment/'):
        with open(filepath + '/train_sentiment/' + filename, 'r') as f:
            sentiment = json.load(f)
        pet_id = filename.split('.')[0]
        sentiment_dict[pet_id] = {}
        sentiment_dict[pet_id]['magnitude'] = sentiment['documentSentiment']['magnitude']
        sentiment_dict[pet_id]['score'] = sentiment['documentSentiment']['score']
        sentiment_dict[pet_id]['language'] = sentiment['language']

    for filename in os.listdir(filepath + '/test_sentiment/'):
        with open(filepath + '/test_sentiment/' + filename, 'r') as f:
            sentiment = json.load(f)
        pet_id = filename.split('.')[0]
        sentiment_dict[pet_id] = {}
        sentiment_dict[pet_id]['magnitude'] = sentiment['documentSentiment']['magnitude']
        sentiment_dict[pet_id]['score'] = sentiment['documentSentiment']['score']
        sentiment_dict[pet_id]['language'] = sentiment['language']

    return train, test, all_data, breeds, colors, states, all_count, sentiment_dict

def data_prepreocessing(train, test, all_data, breeds, colors, sentiment_dict):
    all_data['Type'] = all_data['Type'].apply(lambda x: 'Dog' if x == 1 else 'Cat')
    train['Name'] = train['Name'].fillna('Unnamed')
    test['Name'] = test['Name'].fillna('Unnamed')
    all_data['Name'] = all_data['Name'].fillna('Unnamed')

    train['No_name'] = 0
    train.loc[train['Name'] == 'Unnamed', 'No_name'] = 1
    test['No_name'] = 0
    test.loc[test['Name'] == 'Unnamed', 'No_name'] = 1
    all_data['No_name'] = 0
    all_data.loc[all_data['Name'] == 'Unnamed', 'No_name'] = 1
    train['Pure_breed'] = 0
    train.loc[train['Breed2'] == 0, 'Pure_breed'] = 1
    test['Pure_breed'] = 0
    test.loc[test['Breed2'] == 0, 'Pure_breed'] = 1
    all_data['Pure_breed'] = 0
    all_data.loc[all_data['Breed2'] == 0, 'Pure_breed'] = 1
    breeds_dict = {k: v for k, v in zip(breeds['BreedID'], breeds['BreedName'])}
    train['Breed1_name'] = train['Breed1'].apply(
        lambda x: '_'.join(breeds_dict[x].split()) if x in breeds_dict else 'Unknown')
    train['Breed2_name'] = train['Breed2'].apply(lambda x: '_'.join(breeds_dict[x]) if x in breeds_dict else '-')

    test['Breed1_name'] = test['Breed1'].apply(
        lambda x: '_'.join(breeds_dict[x].split()) if x in breeds_dict else 'Unknown')
    test['Breed2_name'] = test['Breed2'].apply(lambda x: '_'.join(breeds_dict[x].split()) if x in breeds_dict else '-')

    all_data['Breed1_name'] = all_data['Breed1'].apply(
        lambda x: '_'.join(breeds_dict[x].split()) if x in breeds_dict else 'Unknown')
    all_data['Breed2_name'] = all_data['Breed2'].apply(
        lambda x: '_'.join(breeds_dict[x].split()) if x in breeds_dict else '-')
    colors_dict = {k: v for k, v in zip(colors['ColorID'], colors['ColorName'])}
    train['Color1_name'] = train['Color1'].apply(lambda x: colors_dict[x] if x in colors_dict else '')
    train['Color2_name'] = train['Color2'].apply(lambda x: colors_dict[x] if x in colors_dict else '')
    train['Color3_name'] = train['Color3'].apply(lambda x: colors_dict[x] if x in colors_dict else '')

    test['Color1_name'] = test['Color1'].apply(lambda x: colors_dict[x] if x in colors_dict else '')
    test['Color2_name'] = test['Color2'].apply(lambda x: colors_dict[x] if x in colors_dict else '')
    test['Color3_name'] = test['Color3'].apply(lambda x: colors_dict[x] if x in colors_dict else '')

    all_data['Color1_name'] = all_data['Color1'].apply(lambda x: colors_dict[x] if x in colors_dict else '')
    all_data['Color2_name'] = all_data['Color2'].apply(lambda x: colors_dict[x] if x in colors_dict else '')
    all_data['Color3_name'] = all_data['Color3'].apply(lambda x: colors_dict[x] if x in colors_dict else '')

    train['health'] = train['Vaccinated'].astype(str) + '_' + train['Dewormed'].astype(str) + '_' + train[
        'Sterilized'].astype(str) + '_' + train['Health'].astype(str)
    test['health'] = test['Vaccinated'].astype(str) + '_' + test['Dewormed'].astype(str) + '_' + test[
        'Sterilized'].astype(str) + '_' + test['Health'].astype(str)

    train['Free'] = train['Fee'].apply(lambda x: 'Free' if x == 0 else 'Not Free')
    test['Free'] = test['Fee'].apply(lambda x: 'Free' if x == 0 else 'Not Free')
    all_data['Free'] = all_data['Fee'].apply(lambda x: 'Free' if x == 0 else 'Not Free')

    train['Description'] = train['Description'].fillna('')
    test['Description'] = test['Description'].fillna('')
    all_data['Description'] = all_data['Description'].fillna('')

    train['desc_length'] = train['Description'].apply(lambda x: len(x))
    train['desc_words'] = train['Description'].apply(lambda x: len(x.split()))

    test['desc_length'] = test['Description'].apply(lambda x: len(x))
    test['desc_words'] = test['Description'].apply(lambda x: len(x.split()))

    all_data['desc_length'] = all_data['Description'].apply(lambda x: len(x))
    all_data['desc_words'] = all_data['Description'].apply(lambda x: len(x.split()))

    train['averate_word_length'] = train['desc_length'] / train['desc_words']
    test['averate_word_length'] = test['desc_length'] / test['desc_words']
    all_data['averate_word_length'] = all_data['desc_length'] / all_data['desc_words']

    train['lang'] = train['PetID'].apply(lambda x: sentiment_dict[x]['language'] if x in sentiment_dict else 'no')
    train['magnitude'] = train['PetID'].apply(lambda x: sentiment_dict[x]['magnitude'] if x in sentiment_dict else 0)
    train['score'] = train['PetID'].apply(lambda x: sentiment_dict[x]['score'] if x in sentiment_dict else 0)

    test['lang'] = test['PetID'].apply(lambda x: sentiment_dict[x]['language'] if x in sentiment_dict else 'no')
    test['magnitude'] = test['PetID'].apply(lambda x: sentiment_dict[x]['magnitude'] if x in sentiment_dict else 0)
    test['score'] = test['PetID'].apply(lambda x: sentiment_dict[x]['score'] if x in sentiment_dict else 0)

    all_data['lang'] = all_data['PetID'].apply(lambda x: sentiment_dict[x]['language'] if x in sentiment_dict else 'no')
    all_data['magnitude'] = all_data['PetID'].apply(
        lambda x: sentiment_dict[x]['magnitude'] if x in sentiment_dict else 0)
    all_data['score'] = all_data['PetID'].apply(lambda x: sentiment_dict[x]['score'] if x in sentiment_dict else 0)

    return train, test, all_data, breeds, colors

def plot_adoption_speed_distribution(train, all_data):
    plt.figure(figsize=(14, 6));
    g = sns.countplot(x='AdoptionSpeed', data=all_data.loc[all_data['dataset_type'] == 'train'], palette="BuPu")
    plt.title('Adoption speed distribution');
    ax = g.axes
    for p in ax.patches:
        ax.annotate(f"{p.get_height() * 100 / train.shape[0]:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=11, color='gray', rotation=0, xytext=(0, 10),
                    textcoords='offset points')

    plt.figure(figsize=(10, 8));
    labels = '0', '1', '2', '3', '4'
    fraces = [2.73, 20.61, 26.93, 21.74, 27.99]
    explode = [0.05, 0.05, 0.05, 0.05, 0.05]
    plt.axes(aspect=1)
    plt.pie(x=fraces, labels=labels, autopct='%0f%%', explode=explode, shadow=True)
    plt.title('Adoption speed distribution');
    plt.show()


def plot_count_by_feature(dataframe, feature, all_count, hue = 'AdoptionSpeed', title = ''):
    g = sns.countplot(x = feature, data = dataframe, hue = hue, palette="BuPu");
    plt.title(f'AdoptionSpeed {title}');
    ax = g.axes
    all_count = dict(all_count)
    plot_dict = {}
    for i in dataframe[feature].unique():
        feature_count = dict(dataframe.loc[dataframe[feature] == i, 'AdoptionSpeed'].value_counts().sort_index())

        for k, v in all_count.items():
            if k in feature_count:
                plot_dict[feature_count[k]] = ((feature_count[k] / sum(feature_count.values())) / all_count[k]) * 100 - 100
            else:
                plot_dict[0] = 0

    for p in ax.patches:
        h = p.get_height() if str(p.get_height()) != 'nan' else 0
        text = f"{plot_dict[h]:.0f}%" if plot_dict[h] < 0 else f"+{plot_dict[h]:.0f}%"
        ax.annotate(text, (p.get_x() + p.get_width() / 2., h),
             ha='center', va='center', fontsize=11, color='green' if plot_dict[h] > 0 else 'red', rotation=0, xytext=(0, 10),
             textcoords='offset points')

def make_factor_plot(dataframe, feature, col, title, all_count, ann=True, col_wrap=4):
    g = sns.factorplot(col, col = feature, data = dataframe, kind = 'count', col_wrap = col_wrap,palette="BuPu");
    plt.subplots_adjust(top=0.9);
    plt.suptitle(title);
    ax = g.axes
    all_count = dict(all_count)
    plot_dict = {}
    for i in dataframe[feature].unique():
        feature_count = dict(dataframe.loc[dataframe[feature] == i, 'AdoptionSpeed'].value_counts().sort_index())

        for k, v in all_count.items():
            if k in feature_count:
                plot_dict[feature_count[k]] = ((feature_count[k] / sum(feature_count.values())) / all_count[k]) * 100 - 100
            else:
                plot_dict[0] = 0
    if ann:
        for a in ax:
            for p in a.patches:
                text = f"{plot_dict[p.get_height()]:.0f}%" if plot_dict[p.get_height()] < 0 else f"+{plot_dict[p.get_height()]:.0f}%"
                a.annotate(text, (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', fontsize=11, color='green' if plot_dict[p.get_height()] > 0 else 'red', rotation=0, xytext=(0, 10),
                     textcoords='offset points')

def plot_analysis_of_type(all_data, all_count):
    plt.figure(figsize=(18, 8));
    plot_count_by_feature(dataframe=all_data.loc[all_data['dataset_type'] == 'train'], feature='Type',  all_count =  all_count, title='by Type')

def plot_analysis_of_name(all_data, all_count):
    fig, ax = plt.subplots(figsize=(16, 10))
    plt.subplot(1, 2, 1)
    cat_name = ' '.join(all_data.loc[all_data['Type'] == 'Cat', 'Name'].fillna('').values)
    wordcloud_cat = WordCloud(width=1200, height=1000).generate(cat_name)
    plt.imshow(wordcloud_cat)
    plt.title('WordCloud of Cat Name')
    plt.axis("off")

    plt.subplot(1, 2, 2)
    dog_name = ' '.join(all_data.loc[all_data['Type'] == 'Dog', 'Name'].fillna('').values)
    wordcloud_dog = WordCloud(width=1200, height=1000).generate(dog_name)
    plt.imshow(wordcloud_dog)
    plt.title('WordCloud of Dog Name')
    plt.axis("off")
    plt.show()

    plt.figure(figsize=(18, 8));
    plot_count_by_feature(dataframe=all_data.loc[all_data['dataset_type'] == 'train'], feature='No_name', all_count = all_count,
                          title='by whether having a name')
    plt.xticks([0, 1], ['Named', 'Unnamed'])

def plot_analysis_of_age(train):
    plt.figure(figsize=(18, 8));
    sns.boxplot(x="Age", y="AdoptionSpeed", orient='h', data=train);
    plt.title('AdoptionSpeed by age');

    data = []
    for i in range(5):
        df = train.loc[train['AdoptionSpeed'] == i]
        data.append(go.Scatter(
            x=df['Age'].value_counts().sort_index().index,
            y=df['Age'].value_counts().sort_index().values,
            name=str(i)
        ))

    layout = go.Layout(dict(title="AdoptionSpeed trends by Age",
                            xaxis=dict(title='Age (months)'),
                            yaxis=dict(title='Counts'),
                            )
                       )
    py.iplot(dict(data=data, layout=layout), filename='basic-line')

def plot_analysis_of_breed(train, all_count):
    plt.figure(figsize=(18, 8));
    plot_count_by_feature(dataframe=train, feature='Pure_breed', all_count = all_count, title='by whether having pure breed')
    plt.xticks([0, 1], ['Not Pure_breed', 'Pure_breed'])

def plot_analysis_of_gender(train, all_count):
    plt.figure(figsize=(18, 8));
    plot_count_by_feature(dataframe=train, feature='Gender', all_count = all_count, title='by gender')
    plt.xticks([0, 1, 2], ['Male', 'Female', 'Mixed'])

def plot_analysis_of_color(train, all_data, all_count):

    plt.figure(figsize=(18, 8));
    sns.countplot(data=all_data, x='Color1_name',
                  palette=['Black', 'Brown', '#FFFDD0', 'Gray', 'Gold', 'White', 'Yellow']);
    plt.title('Counts of pets in datasets by main color');

    make_factor_plot(dataframe=train, feature='Color1_name', col='AdoptionSpeed',
                     title='Counts of pets by main color and Adoption Speed', all_count = all_count);


def plot_analysis_of_matiritysize(train, all_count):
    plt.figure(figsize=(18, 8));
    plot_count_by_feature(dataframe=train, feature='MaturitySize', all_count = all_count, title='by maturitySize')
    plt.xticks([0, 1, 2, 3], ['Small', 'Medium', 'Large', 'Extra Large'])

def plot_analysis_of_health(train, all_count):
    plt.figure(figsize=(20, 12));
    plt.subplot(2, 2, 1)
    plot_count_by_feature(dataframe=train, feature='Vaccinated', all_count = all_count, title='by whether vaccinated')
    plt.xticks([0, 1, 2], ['Yes', 'No', 'Not sure']);
    plt.title('AdoptionSpeed and Vaccinated');

    plt.subplot(2, 2, 2)
    plot_count_by_feature(dataframe=train, feature='Dewormed', all_count = all_count, title='by whether dewormed')
    plt.xticks([0, 1, 2], ['Yes', 'No', 'Not sure']);
    plt.title('AdoptionSpeed and Dewormed');

    plt.subplot(2, 2, 3)
    plot_count_by_feature(dataframe=train, feature='Sterilized', all_count = all_count, title='by whether sterilized')
    plt.xticks([0, 1, 2], ['Yes', 'No', 'Not sure']);
    plt.title('AdoptionSpeed and Sterilized');

    plt.subplot(2, 2, 4)
    plot_count_by_feature(dataframe=train, feature='Health', all_count = all_count, title='by Health')
    plt.xticks([0, 1, 2], ['Healthy', 'Minor Injury', 'Serious Injury']);
    plt.title('AdoptionSpeed and Health');

    plt.suptitle('Adoption Speed and health conditions');

def plot_analysis_of_fee(train, all_count):
    plt.figure(figsize=(18, 8));
    plot_count_by_feature(dataframe=train, feature='Free', all_count = all_count, title='by whether free')

def main():
    train, test, all_data, breeds, colors, states, all_count, sentiment_dict = read_data('data')
    train, test, all_data, breeds, colors = data_prepreocessing(train, test, all_data, breeds, colors, sentiment_dict)
    plot_adoption_speed_distribution(train, all_data)
    plot_analysis_of_type(all_data, all_count)
    plot_analysis_of_name(all_data, all_count)
    plot_analysis_of_age(train)
    plot_analysis_of_breed(train, all_count)
    plot_analysis_of_gender(train, all_count)
    plot_analysis_of_color(train, all_data, all_count)
    plot_analysis_of_matiritysize(train, all_count)
    plot_analysis_of_health(train, all_count)
    plot_analysis_of_fee(train, all_count)

if __name__ == '__main__':
	main()
