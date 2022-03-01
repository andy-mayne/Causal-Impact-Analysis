
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import causalimpact 
import tensorflow

st.set_page_config(page_title='Causal Inference',
                    layout="wide", 
                    page_icon='📈'
                    )

st.image("images/1.png", 
        width=600
        )

st.markdown("## Causal Impact Analysis")

st.markdown('''This tool can be used to work out the possible causal effect of an intervention on a time series. 
For example, how many more patient discharges can we acheive since changing the patient pathway? Answering a question like this can be difficult when a randomised experiment is not possible. 
Causal Impact Analysis aims to address this by estimating what would have happened after the intervention if the intervention had not occurred.

As with all approaches to causal inference on non-experimental data, valid conclusions require strong assumptions. Causal Impact in particular, assumes that the outcome can be explained by something not 
affected by the intervention. Furthermore, the relation between treated series and control series is assumed to be stable during the post-intervention period. Understanding and checking these assumptions for any given application is critical for obtaining valid conclusions.''')

uploaded_file = st.file_uploader(label='Upload your data to get started', 
                                type=('.csv', '.xlsx'), 
                                help='Data should be stored in an csv or excel spreadsheet, the first column should be the date column, the rest should be numerical.'
                                )

c1, c2, c3, c4 = st.columns((1,1,1,1))

if uploaded_file != None:

    with st.spinner('Uploading your data'):
            if '.xlsx' in uploaded_file.name:
                df = pd.read_excel(uploaded_file)
            elif '.csv' in uploaded_file.name:
                df = pd.read_csv(uploaded_file)
            else:
                c1.warning('Sorry there is no data to upload, or it is in the wrong format, make sure its a .csv or .xlsx file!')

if uploaded_file != None:
    
    indexedcolumn = c1.selectbox(label="Date Column",
                                 options = df.select_dtypes(include=['datetime64']).columns.values.tolist(), 
                                 help = 'If nothing is showing then you may need to go back to your upload and format the column to a date column.'
                                 )
    targetcolumn = c2.selectbox(label="Target Column", 
                                options = df.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns.values.tolist(), 
                                help='This is the column that you are trying to measure the change on.'
                                )
    covariatecolumn = c3.selectbox(label="Covariate", 
                                options = df.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns.values.tolist(), 
                                help='This is the column that represents the control (the one that was not impacted by the change).'
                                )
    
    df[indexedcolumn] = pd.to_datetime(df[indexedcolumn])
 
    startchange = c4.date_input(label='Intervention Date', 
                                min_value = min(df[indexedcolumn]), 
                                max_value = max(df[indexedcolumn]), 
                                value = max(df[indexedcolumn])+pd.DateOffset(-30), 
                                help='The intervention was the date you made the change'
                                )

    r1,r2,r3,r4 = st.columns((1,1,1,1))

    onlychangeperiod = r1.checkbox('Only Show Change Period')
    advancedtoggle = r2.checkbox('Change default model')


    if advancedtoggle:
        a1, a2, a3, a4, a5, a6 = st.columns((1,1,1,1,1,1))


        mcmcsamples = a1.number_input(label='MCMC Samples to Draw From', 
                                        value=1000, 
                                        min_value=0, 
                                        max_value=5000, 
                                        help='Number of MCMC samples to draw. More samples lead to more accurate inferences. Defaults to 1000.'
                                        )
        standardise = a2.selectbox(label='Standardise Data', 
                                    index=0, 
                                    options=(True, False), 
                                    help='Whether to standardize all columns of the data before fitting the model. This is equivalent to an empirical Bayes approach to setting the priors. It ensures that results are invariant to linear transformations of the data'
                                    )
        prior_sd = a3.selectbox(label='Prior Standard Deviation ', 
                                options=(0.01, 0.1), 
                                index=0, 
                                help='Prior standard deviation of the Gaussian random walk of the local level. Expressed in terms of data standard deviations.'
                                )
        nseasonseaons = a4.number_input(label='Seasonal Component', 
                                value=1,
                                min_value=1, 
                                max_value=365,
                                help='Period of the seasonal components'
                                )
        seasonalduration = a5.number_input(label='Duration of each season', 
                                value=1,
                                min_value=1, 
                                max_value=365,
                                help='Period of the seasonal components. For example, to add a day-of-week component to the data with daily granularity, Component = 7, Duration = 1). To add a day-of-week component to data with hourly granularity, Component = 7, Duration = 24).'
                                )
        dynamicregression = a6.selectbox(label='Standardise Data', 
                                    index=1, 
                                    options=(True, False), 
                                    help='Whether to include time-varying regression coefficients. In combination with a time-varying local trend or even a time-varying local level, this often leads to overspecification, in which case a static regression is safer.'
                                    )

    else:
        mcmcsamples = 1000
        standardise = True
        prior_sd = 0.01
        nseasonseaons = 1
        seasonalduration = 1
        dynamicregression = False

    startchange = pd.to_datetime(startchange)

    pre_period = [min(df[indexedcolumn]), startchange + pd.DateOffset(-1)]
    post_period = [startchange,max(df[indexedcolumn])]

    df = df[[indexedcolumn,targetcolumn, covariatecolumn]].set_index(indexedcolumn)

    st.markdown('#### Time Series of ' + targetcolumn + ' & ' + covariatecolumn)
    st.markdown('''This graph shows the data that has been uploaded and the chosen target and covariate fields. You should check to ensure that the covariate 
    (orange line) follows the same pattern as the target (blue line), but is **not** impacted by the change you are trying to measure marked in grey.''')

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(x=df.index, y=df[targetcolumn],
                    mode='lines',
                    name=targetcolumn, 
                    line=dict(color="#44546A")
                    )) 

    fig2.add_trace(go.Scatter(x=df.index, y=df[covariatecolumn],
                    mode='lines',
                    name=covariatecolumn,
                    line=dict(color="#ED7D31")
                    )) 

    fig2.update_layout(height = 400, 
                    margin=dict(r=1, l=1, t=1, b=1),
                    xaxis_title=indexedcolumn, 
                    yaxis_title=targetcolumn, 
                    legend=dict(yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                ))

    fig2.add_vrect(x0=startchange, x1=max(df.index), 
                annotation_text="Change Period", 
                annotation_position="top left",
                fillcolor="black", 
                opacity=0.25, 
                line_width=0
                )

    st.plotly_chart(fig2, use_container_width=True)

    impact = causalimpact.CausalImpact(df, 
                                        pre_period, 
                                        post_period, 
                                        model_args={"niter":mcmcsamples, 
                                        "nseasons":nseasonseaons,
                                        "standardize_data":standardise,
                                        "prior_level_sd":prior_sd,
                                        "season_duration":seasonalduration,
                                        "dynamic_regression":dynamicregression}
                                        )

    impact.run()

    originalfig = impact.plot('original')
    pointwisefig = impact.plot('pointwise')
    cumulativefig = impact.plot('cumulative')
    summaryinfo = impact.summary()
    detailedreport = impact.summary("report")

    raw = impact.inferences.to_csv().encode('utf-8')

    if onlychangeperiod:

        originalfig.update_xaxes(range=(startchange,max(df.index)))
        pointwisefig.update_xaxes(range=(startchange,max(df.index)))
        cumulativefig.update_xaxes(range=(startchange,max(df.index)))

    st.markdown('#### Original & Counterfactual')
    st.markdown('''This graph compares what actually happened (blue line) and what we would have predicted to happen (**counterfactual orange line**). 
                    In light grey you can see the prediction intervals. The change date is marked with a red dotted line''')

    st.plotly_chart(originalfig, 
                    use_container_width=True)
    
    st.markdown('#### Pointwise Causal Effect')
    st.markdown('''This graph is showing the **pointwise** which is the difference between what actually happened, and what was predicted to happen''')

    st.plotly_chart(pointwisefig, 
                    use_container_width=True)

    st.markdown('#### Cumulative impact of change')
    st.markdown('''This graph is showing the **cumulative impact** of the change''')

    st.plotly_chart(cumulativefig, 
                    use_container_width=True)

    st.markdown('#### Test Results')
    st.download_button('Download Raw Results', raw, file_name='CausalImpactAnalysis.csv')   

    if summaryinfo != None:
        
        st.markdown('Summary Information')
        st.write(summaryinfo)

        st.markdown('Detailed Information')
        st.write(detailedreport)

with st.expander('Source', expanded=False):
    
    st.markdown('''Emergency predict is an open source library of Data Science tools for those in the public sector to use  \n  \n This particular tool has 
    been adapted from the causal impact python library: http://google.github.io/CausalImpact/ which was moved over to python 
    by: https://github.com/jamalsenouci  \n  \n For a great explanation on what causal impact analysis is please watch the below 
    video by Kay Broderson at the Big Things Conference in 2016''')

    st.video(data='https://www.youtube.com/watch?v=GTgZfCltMm8')


    
