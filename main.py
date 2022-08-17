import random
import math
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from scipy.stats import lognorm
from scipy.stats import norm
import plotly.graph_objects as go
from scipy.stats import expon

# Input assumptions

# Flow rate
l0 = (1343000/24) #basic flow rate [transactions/hour]
transactions_volatility_b = 0.5 #block volatility - normal distribution sigma
min_rate = 10000 #minimal transaction hourly rate

#Subcluster
n_sc = 11 #total subcluster share - 1 main cluster, and n_sc-1 subclusters
main_cluster_share = 0.5 #share of main cluster in transaction flow

#Gas price
gas_price_base = 15 #Inital gas price [gwei]
el = 0.2 #elasticity coefficient
min_gas_price = 5.0 #minimum gas price
#priority_fee_median = 5 #priority fee median
#priority_lognormal_sigma = 0.5 #priority fee lognormal sigma
normal_transaction_time = 5*60 #seconds
transaction_lognormal_sigma = 0.5 #seconds

#Blocks
parent_gas_target = 15*10**6 #target block size in gas
parent_gas_limit = parent_gas_target*2 #block  limit
bt = 15 #block generation time in seconds

#Transaction size
gu_base = 80000 #avg. transaction size in gas
gu_volatility_2 = 0.3 #transaction size volatility

#Proof verification and generation
CPUH = 8.4 #proof generation cost in CPU-hour
CPUH_price = 0.134012 #CPU-hour cost in USD https://cloud.google.com/compute/all-pricing
proof_generation = CPUH*CPUH_price
proof_verification = 2*10**6 #proof verification cost in gas


#Other
stuck_th = 3600 #Period in seconds in mempool we assume, that transaction is stuck
ETH_price = 1700 #USD

block_id = np.zeros(n_sc+1)
gas_price = np.zeros(n_sc+1)
gas_price.fill(gas_price_base)
user_price = np.zeros(n_sc+1)
user_price.fill(gas_price_base)
lc = np.zeros(n_sc+1)

#calculate gas price eip-1559 https://github.com/ethereum/EIPs/blob/master/EIPS/eip-1559.md
def current_gas_price(gas_price, parent_gas_used):
    gas_used_delta = parent_gas_used - parent_gas_target
    val = max(min_gas_price, gas_price*(1 + (gas_used_delta/parent_gas_target)*0.125))
    return val

#calculate wait time for succesfull transactions (not in mempool)
#if wait time > 1h, than we skip it from wait time calc, assumping that transaction stucks in mempool
def wait_time(row):
    if row['status']=='pending':
        return np.nan
    else:
        wt = (row['processed_time'] - row['arrival_time'])
        if (wt<stuck_th):
            return wt
        else:
            return np.nan

#calc median for lognormal user bid distribution - avg. user price for last hour transactions with wait time < 15 min
def calc_user_price(c):
    sc_mempool = mempool[mempool['sc']==c][['arrival_time','gas_unit','user_price']]
    lh_flow = pd.concat([sc_mempool[sc_mempool['arrival_time']>=current_time-60][['gas_unit','user_price']],lh_blocks[lh_blocks['sc']==c][['gas_unit','user_price']]], ignore_index = True)
    vg = min_gas_price
    while True:
        t = sc_mempool[sc_mempool['user_price']>=vg]['gas_unit'].sum()/(parent_gas_target/bt - (len(lh_flow[lh_flow['user_price']>=vg])/len(lh_flow))*(lc[c]/3600)*gu_base)
        if (t<=normal_transaction_time)&(t>=0):
            break
        vg = vg + 2
    return vg


# block generation function
def transaction_process(df, sc):
    df_tmp = df.loc[(df['user_price'] >= gas_price[sc]) & (
                df['sc'] == sc)]  # select transaction with user price > gas price within selected cluster
    df_tmp = df_tmp.sort_values(by=['user_price'])  # prioritize transacton by priority fee
    df_tmp['cum_gas_sum'] = df_tmp['gas_unit'].cumsum()  # calculate gas usage cummulative sum

    df_out = df_tmp.loc[df_tmp[
                            'cum_gas_sum'] <= parent_gas_limit]  # if gas usage > gas limit, skip transactions with lowerst priority fee
    if (len(df_out) > 0) & (
            sc > 0):  # check if block not empty and subcluster - calculate number of transaction within a block
        add_fee = (proof_generation + proof_verification * gas_price[sc] * ETH_price * 10 ** -9) / len(df_out)
    else:
        add_fee = 0
    df_out['block_id'] = block_id[sc]  # set block id
    df_out['status'] = 'success'  # change status
    df_out['gas_price'] = gas_price[sc]  # write down actual gas price
    df_out['act_fee'] = df_out['user_price'] * df_out[
        'gas_unit'] * ETH_price * 10 ** -9 + add_fee  # calc actual fee in USD + add_fee - cost of verification in case of subcluster
    df_out['processed_time'] = current_time  # write down processed time
    return df_out, pd.concat([df.loc[(df['user_price'] < gas_price[sc]) | (df['sc'] != sc)],
                              df_tmp.loc[df_tmp['cum_gas_sum'] > parent_gas_limit]], ignore_index=True)


# generate transactions flow
def transactions_flow(lr, sc):
    l = lr * min(1, (gas_price_base / gas_price[sc]) ** el)  # cutting demand with elasticity

    if sc == 0:
        l = l  # if cluster == 0, i.e. bulk model, than rate not change
    # if sc==1 - main subcluster, sc>1 other subclusters
    elif sc == 1:
        l = l * main_cluster_share
    else:
        l = ((1 - main_cluster_share) / (n_sc - 1)) * l

    lc[sc] = l
    tick_time = 0
    data = []
    while tick_time < bt:
        inter_arrival_time = (-math.log(
            1.0 - random.random()) / l) * 3600  # calc arrival interval with Poisson distribution
        tick_time = tick_time + inter_arrival_time
        if tick_time >= bt:
            break
        gu_t = norm.ppf(random.random(), loc=gu_base, scale=l0 * gu_volatility_2)  # cal transaction gas usage
        user_price_n = expon.ppf(random.random(), loc=user_price[sc],
                                 scale=transaction_lognormal_sigma * user_price[sc])
        data.append({
            'arrival_time': current_time + tick_time,
            'user_price': user_price_n,
            # 'priority_fee': priority_fee,
            'gas_price': gas_price[sc],
            'gas_unit': gu_t,
            'status': 'pending',
            'processed_time': np.nan,
            'act_fee': np.nan,
            'block_id': 0,
            'sc': sc
        })
    return data

if __name__ == "__main__":
    current_time = 0
    h = 0

    df = pd.DataFrame()
    mempool = pd.DataFrame()
    blocks = pd.DataFrame()
    lh_blocks = pd.DataFrame()
    h = 1

    l = max(10000, norm.ppf(random.random(), loc=l0, scale=l0 * transactions_volatility_b))
    m = 0
    m15 = 0
    m_stuck = []
    for b in np.arange(1, int(12 * 3600 / bt), 1):
        # generate transaction flows by cluster
        for c in np.arange(0, n_sc + 1, 1):
            d = transactions_flow(l, c)
            mempool = pd.concat([mempool, pd.DataFrame(d)], ignore_index=True)
        current_time = current_time + bt

        # generate bocks by cluster
        for c in np.arange(0, n_sc + 1, 1):
            block_id[c] = block_id[c] + 1
            new_block, mempool = transaction_process(mempool, c)
            blocks = pd.concat([blocks, new_block], ignore_index=True)
            lh_blocks = pd.concat([lh_blocks, new_block[['arrival_time', 'gas_unit', 'user_price', 'sc']]],
                                  ignore_index=True)
            parent_gas_used = new_block['gas_unit'].sum()
            gas_price[c] = current_gas_price(gas_price[c], parent_gas_used)
            # if c == 0:
            # print("Gas price:"+str(round(gas_price[0])) + " Rate: "+str(round(l,0))+" mempool: "+str(len(mempool.loc[mempool['sc']==0])))
        lh_blocks = lh_blocks[
            lh_blocks['arrival_time'] >= current_time - 3600]  # database of last hour blocks for calc user bids

        # each 15 minutes change flow rate by normal distribution
        if (current_time - m15) / 60 > 15:
            m15 = current_time
            l = max(10000, norm.ppf(random.random(), loc=l0, scale=l0 * transactions_volatility_b))

        # each minute calculate number of stucked stransactions and recalculate avg. user price with wait time < 15 min
        if (current_time - m) / 60 > 1:
            m = current_time
            df_tmp = mempool[(mempool['arrival_time'] < current_time - stuck_th)]
            m_stuck.append(
                {
                    'm': math.floor(m / 60),
                    'model': 'single cluster',
                    'stuck': len(df_tmp[df_tmp['sc'] == 0])
                }
            )
            m_stuck.append(
                {
                    'm': math.floor(m / 60),
                    'model': 'multi clusters',
                    'stuck': len(df_tmp[df_tmp['sc'] > 0])
                }
            )
            for c in np.arange(0, n_sc + 1, 1):
                user_price[c] = calc_user_price(c)
            print("minute: " + str(math.floor(m / 60)) + " Gas price [gwei]:" + str(round(gas_price[0])) + " Minimum bid gas price [gwei]:" + str(
                round(user_price[0])) + " Transactions flow [#/h]: " + str(round(l, 0)) + " Mempool size: " + str(
                len(mempool.loc[mempool['sc'] == 0])))

    # combine all data in one dataframe for charts
    df_all = pd.concat([blocks, mempool], ignore_index=True)
    df_all['m'] = df_all['arrival_time'].apply(lambda x: math.floor(x / 60))
    df_all['wait_time'] = df_all.apply(lambda row: wait_time(row), axis=1)
    df_all['model'] = df_all['sc'].apply(lambda x: 'single cluster' if x == 0 else 'multi clusters')
    df_all['stuck'] = df_all['status'].apply(lambda x: 1 if x == 'pending' else 0)
    df_res = df_all.groupby(["m", "model"]).agg(
        act_fee=pd.NamedAgg(column="act_fee", aggfunc="mean"),
        wait_time=pd.NamedAgg(column="wait_time", aggfunc="mean"),
        transaction_cnt=pd.NamedAgg(column="status", aggfunc="count"),
        gas_price=pd.NamedAgg(column="gas_price", aggfunc="mean")
    ).reset_index()
    # df_res['t_count'] = 1
    # df_res['cum_t_count'] = df_res['t_count'].cumsum()

    df_tmp = pd.DataFrame(m_stuck)

    df_t = df_all[['m', 'model', 'status']].groupby(['m', 'model']).count().reset_index().pivot(index="m",
                                                                                                columns=["model"],
                                                                                                values="status").reset_index()
    for v in ['single cluster', 'multi clusters']:
        df_t[v] = df_t[v].cumsum()

    df_t = df_t.melt(
        id_vars="m",
        value_name="Value")

    df_tmp = df_tmp.merge(df_t, left_on=['m', 'model'], right_on=['m', 'model'])
    df_tmp['stuck_share'] = round((df_tmp['stuck'] / df_tmp['Value']) * 100, 1)

    df_total = df_all.groupby(["model"]).agg(
        act_fee=pd.NamedAgg(column="act_fee", aggfunc="mean"),
        wait_time=pd.NamedAgg(column="wait_time", aggfunc="mean"),
        transaction_cnt=pd.NamedAgg(column="status", aggfunc="count"),
        gas_price=pd.NamedAgg(column="gas_price", aggfunc="mean")
    ).reset_index().rename(columns={"act_fee": "Avg. transaction fee [$]", "wait_time": "Avg. wait time [sec]",
                                    "transaction_cnt": "Number of transactions", "gas_price": "Avg. gas price [gwei]"})

    df_total["Avg. transaction fee [$]"] = df_total["Avg. transaction fee [$]"].apply(lambda x: round(x, 1))
    df_total["Avg. wait time [sec]"] = df_total["Avg. wait time [sec]"].apply(lambda x: round(x, 0))
    df_total["Number of transactions"] = df_total["Number of transactions"].apply(lambda x: round(x, 0))
    df_total["Avg. gas price [gwei]"] = df_total["Avg. gas price [gwei]"].apply(lambda x: round(x, 0))

    #generate charts
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=5, cols=1,
        row_heights=[0.2, 0.2, 0.2, 0.2, 0.2],
        specs=[
                [{"secondary_y": True}],
                [{"secondary_y": True}],
                [{"secondary_y": True}],
                [{"secondary_y": True}],
                [{"type": "table"}],
              ],
        subplot_titles=("Gas price [gwei]", "Avg. transaction fee [$/#]", "Avg. wait time [sec]", 'Stuck in mempool [% of transactions]')
    )

    cols = {
      'r': 'rgb(245, 112, 101)',
      'g': 'rgb(82, 217, 200)',
      'db': 'rgb(37, 45, 100)',
      'v': 'rgb(143, 143, 191)',
      'lb': 'rgb(124, 200, 236)',
      'o': 'rgb(248, 211, 94)',
      'gr': 'rgb(175, 171, 171)',
      'b': 'rgb(0, 0, 0)'
    }

    aCols ={
        "multi clusters": cols['lb'],
        'single cluster': cols['r']
    }

    w = 2

    field = 'gas_price'
    model = 'single cluster'
    dash = 'solid'
    sec_y = False
    col = 1
    row = 1
    legend_group = '1'
    opacity = 1.0
    fig.add_trace(
                    go.Scatter( x=df_res.loc[df_res['model']==model]['m'],
                                y=df_res.loc[df_res['model']==model][field],
                                line=dict(color=aCols[model], width=w, dash = dash),
                                legendgroup = legend_group,
                                opacity = opacity,
                                name=model + ': ' + field),
                                col = col,
                                row = row,
                                secondary_y = sec_y
    )

    field = 'gas_price'
    model = 'multi clusters'
    dash = 'solid'
    sec_y = False
    col = 1
    row = 1
    opacity = 1.0
    fig.add_trace(
                    go.Scatter( x=df_res.loc[df_res['model']==model]['m'],
                                y=df_res.loc[df_res['model']==model][field],
                                line=dict(color=aCols[model], width=w, dash = dash),
                                legendgroup = legend_group,
                                opacity = opacity,
                                name=model + ': ' + field),
                                col = col,
                                row = row,
                                secondary_y = sec_y
    )

    field = 'transaction_cnt'
    model = 'single cluster'
    dash = 'dot'
    sec_y = True
    col = 1
    row = 1
    opacity = 0.5
    fig.add_trace(
                    go.Scatter( x=df_res.loc[df_res['model']==model]['m'],
                                y=df_res.loc[df_res['model']==model][field],
                                line=dict(color=aCols[model], width=w, dash = dash),
                                legendgroup = legend_group,
                                opacity = opacity,
                                name=model + ': ' + field),
                                col = col,
                                row = row,
                                secondary_y = sec_y
    )

    field = 'transaction_cnt'
    model = 'multi clusters'
    dash = 'dot'
    sec_y = True
    col = 1
    row = 1
    opacity = 0.5
    fig.add_trace(
                    go.Scatter( x=df_res.loc[df_res['model']==model]['m'],
                                y=df_res.loc[df_res['model']==model][field],
                                line=dict(color=aCols[model], width=w, dash = dash),
                                legendgroup = legend_group,
                                opacity = opacity,
                                name=model + ': ' + field),
                                col = col,
                                row = row,
                                secondary_y = sec_y
    )

    #-----------------------------------------------

    field = 'act_fee'
    model = 'single cluster'
    dash = 'solid'
    sec_y = False
    col = 1
    row = 2
    opacity = 1.0
    legend_group = '2'
    fig.add_trace(
                    go.Scatter( x=df_res.loc[df_res['model']==model]['m'],
                                y=df_res.loc[df_res['model']==model][field],
                                line=dict(color=aCols[model], width=w, dash = dash),
                                legendgroup = legend_group,
                                opacity = opacity,
                                name=model + ': ' + field),
                                col = col,
                                row = row,
                                secondary_y = sec_y
    )
    field = 'act_fee'
    model = 'multi clusters'
    dash = 'solid'
    sec_y = False
    col = 1
    row = 2
    opacity = 1.0
    fig.add_trace(
                    go.Scatter( x=df_res.loc[df_res['model']==model]['m'],
                                y=df_res.loc[df_res['model']==model][field],
                                line=dict(color=aCols[model], width=w, dash = dash),
                                legendgroup = legend_group,
                                opacity = opacity,
                                name=model + ': ' + field),
                                col = col,
                                row = row,
                                secondary_y = sec_y
    )

    field = 'transaction_cnt'
    model = 'single cluster'
    dash = 'dot'
    sec_y = True
    col = 1
    row = 2
    opacity = 0.5
    fig.add_trace(
                    go.Scatter( x=df_res.loc[df_res['model']==model]['m'],
                                y=df_res.loc[df_res['model']==model][field],
                                line=dict(color=aCols[model], width=w, dash = dash),
                                legendgroup = legend_group,
                                opacity = opacity,
                                name=model + ': ' + field),
                                col = col,
                                row = row,
                                secondary_y = sec_y
    )

    field = 'transaction_cnt'
    model = 'multi clusters'
    dash = 'dot'
    sec_y = True
    col = 1
    row = 2
    opacity = 0.5
    fig.add_trace(
                    go.Scatter( x=df_res.loc[df_res['model']==model]['m'],
                                y=df_res.loc[df_res['model']==model][field],
                                line=dict(color=aCols[model], width=w, dash = dash),
                                legendgroup = legend_group,
                                opacity = opacity,
                                name=model + ': ' + field),
                                col = col,
                                row = row,
                                secondary_y = sec_y
    )

    #-----------------------------------------------

    field = 'wait_time'
    model = 'single cluster'
    dash = 'solid'
    sec_y = False
    col = 1
    row = 3
    opacity = 1.0
    legend_group = '3'
    fig.add_trace(
                    go.Scatter( x=df_res.loc[df_res['model']==model]['m'],
                                y=df_res.loc[df_res['model']==model][field],
                                line=dict(color=aCols[model], width=w, dash = dash),
                                legendgroup = legend_group,
                                opacity = opacity,
                                name=model + ': ' + field),
                                col = col,
                                row = row,
                                secondary_y = sec_y
    )

    field = 'wait_time'
    model = 'multi clusters'
    dash = 'solid'
    sec_y = False
    col = 1
    row = 3
    opacity = 1.0
    fig.add_trace(
                    go.Scatter( x=df_res.loc[df_res['model']==model]['m'],
                                y=df_res.loc[df_res['model']==model][field],
                                line=dict(color=aCols[model], width=w, dash = dash),
                                legendgroup = legend_group,
                                opacity = opacity,
                                name=model + ': ' + field),
                                col = col,
                                row = row,
                                secondary_y = sec_y
    )

    field = 'transaction_cnt'
    model = 'single cluster'
    dash = 'dot'
    sec_y = True
    col = 1
    row = 3
    opacity = 0.5
    fig.add_trace(
                    go.Scatter( x=df_res.loc[df_res['model']==model]['m'],
                                y=df_res.loc[df_res['model']==model][field],
                                line=dict(color=aCols[model], width=w, dash = dash),
                                legendgroup = legend_group,
                                opacity = opacity,
                                name=model + ': ' + field),
                                col = col,
                                row = row,
                                secondary_y = sec_y
    )

    field = 'transaction_cnt'
    model = 'multi clusters'
    dash = 'dot'
    sec_y = True
    col = 1
    row = 3
    opacity = 0.5
    fig.add_trace(
                    go.Scatter( x=df_res.loc[df_res['model']==model]['m'],
                                y=df_res.loc[df_res['model']==model][field],
                                line=dict(color=aCols[model], width=w, dash = dash),
                                legendgroup = legend_group,
                                opacity = opacity,
                                name=model + ': ' + field),
                                col = col,
                                row = row,
                                secondary_y = sec_y
    )

    #-----------------------------------------------

    field = 'stuck_share'
    model = 'single cluster'
    dash = 'solid'
    sec_y = False
    col = 1
    row = 4
    opacity = 1.0
    legend_group = '4'
    fig.add_trace(
                    go.Scatter( x=df_tmp.loc[df_tmp['model']==model]['m'],
                                y=df_tmp.loc[df_tmp['model']==model][field],
                                line=dict(color=aCols[model], width=w, dash = dash),
                                legendgroup = legend_group,
                                opacity = opacity,
                                name=model + ': ' + field),
                                col = col,
                                row = row,
                                secondary_y = sec_y
    )

    field = 'stuck_share'
    model = 'multi clusters'
    dash = 'solid'
    sec_y = False
    col = 1
    row = 4
    opacity = 1.0
    fig.add_trace(
                    go.Scatter( x=df_tmp.loc[df_tmp['model']==model]['m'],
                                y=df_tmp.loc[df_tmp['model']==model][field],
                                line=dict(color=aCols[model], width=w, dash = dash),
                                legendgroup = legend_group,
                                opacity = opacity,
                                name=model + ': ' + field),
                                col = col,
                                row = row,
                                secondary_y = sec_y
    )

    field = 'transaction_cnt'
    model = 'single cluster'
    dash = 'dot'
    sec_y = True
    col = 1
    row = 4
    opacity = 0.5
    fig.add_trace(
                    go.Scatter( x=df_res.loc[df_res['model']==model]['m'],
                                y=df_res.loc[df_res['model']==model][field],
                                line=dict(color=aCols[model], width=w, dash = dash),
                                legendgroup = legend_group,
                                opacity = opacity,
                                name=model + ': ' + field),
                                col = col,
                                row = row,
                                secondary_y = sec_y
    )

    field = 'transaction_cnt'
    model = 'multi clusters'
    dash = 'dot'
    sec_y = True
    col = 1
    row = 4
    opacity = 0.5
    fig.add_trace(
                    go.Scatter( x=df_res.loc[df_res['model']==model]['m'],
                                y=df_res.loc[df_res['model']==model][field],
                                line=dict(color=aCols[model], width=w, dash = dash),
                                opacity = opacity,
                                legendgroup = legend_group,
                                name=model + ': ' + field),
                                col = col,
                                row = row,
                                secondary_y = sec_y
    )

    col = 1
    row = 5
    fig.add_trace(
                    go.Table(
                        header=dict(values=list(df_total.columns),
                                    fill_color=cols['db'],
                                    height=30,
                                    align='left'),
                        cells=dict(values=df_total.transpose().values.tolist(),
                                   fill_color='rgb(17,17,17)',
                                   height=30,
                                   align='left')
                        ),
                        col = col,
                        row = row,
    )

    fig.update_layout(
        height=2500,
        template='plotly_dark'
    )
    fig.update_layout(legend_tracegroupgap = 450)
    fig.update_layout(legend=dict(groupclick="toggleitem"))
    fig.update_layout(
        yaxis2=dict(
            showgrid = False
        ),
        yaxis4=dict(
            showgrid = False
        ),
        yaxis6=dict(
            showgrid = False
        ),
        yaxis8=dict(
            showgrid = False
        )
    )

    fig.write_html('results.html')
