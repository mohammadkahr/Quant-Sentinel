"""Offline invariants: no model calls, live feeds, keys or original study data required."""
import asyncio
import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from benchmark.baseline_quant import baseline_quant
from benchmark.statistics import paired_analysis, report_run
from benchmark.audit import fleiss_binary, ordinal_alpha, holm
from core.config import AppConfig, settings
from core.schemas import Decision
from database.store import EventStore
from execution.triple_barrier import TripleBarrierPosition, vwap
from risk.risk_engine import RiskEngine
from tools.telemetry import normalize_book, indicators


def book(price=100., now=0., spread=.02):
    return normalize_book({"bids": [[price-spread/2-i*.01, 10000] for i in range(20)],
                           "asks": [[price+spread/2+i*.01, 10000] for i in range(20)]}, now)


class RiskTests(unittest.TestCase):
    def test_equations_and_caps_across_volatility(self):
        for direction in ("LONG", "SHORT"):
            for atr in (0., .01, .2, 1., 5., 20.):
                p = RiskEngine.calculate_trade_parameters(direction, 100., atr, 10000., 1.5, 10)
                self.assertAlmostEqual(abs(p['entry_price']-p['stop_loss']), max(1.5*atr, .5))
                self.assertAlmostEqual(abs(p['take_profit']-100), 2*abs(p['stop_loss']-100))
                self.assertLessEqual(p['risk_usd'], 150. + 1e-8)
                self.assertLessEqual(p['margin_usd'], 1500. + 1e-8)
                self.assertLessEqual(p['leverage'], 10)
                self.assertAlmostEqual(p['notional_size_usd'], p['coin_amount']*100)

    def test_invalid_inputs_and_no_trade(self):
        for args in [('BUY',100,1), ('LONG',0,1), ('LONG',100,-1), ('LONG',math.nan,1)]:
            with self.assertRaises(ValueError):
                RiskEngine.calculate_trade_parameters(*args)
        with self.assertRaises(ValueError):
            RiskEngine.calculate_trade_parameters('LONG',100,1,account_balance=0)
        self.assertEqual(RiskEngine.calculate_trade_parameters('NO-TRADE',100,1)['notional_size_usd'],0)


class BaselineTests(unittest.TestCase):
    def signal(self, rsi, price=100., low=90., high=110., ema=99.):
        return baseline_quant.evaluate({'current_price_usd':price,'timeframe_analysis':{
            '15m_execution':dict(rsi=rsi,bb_low=low,bb_high=high,ema_20=ema)}})

    def test_rsi_boundaries(self):
        for rsi, expected in [(55,'NO-TRADE'),(55.01,'LONG'),(65,'LONG'),(65.01,'NO-TRADE'),(75,'NO-TRADE')]:
            self.assertEqual(self.signal(rsi)['direction'],expected)
        self.assertEqual(self.signal(70,high=100)['direction'],'SHORT')
        self.assertEqual(self.signal(32,low=100)['direction'],'LONG')
        self.assertEqual(self.signal(33,low=100)['direction'],'NO-TRADE')
        self.assertEqual(self.signal(60)['tokens_used'],0)


class ExecutionTests(unittest.TestCase):
    def position(self, direction='LONG', hold=45, funding=.0008):
        plan=RiskEngine.calculate_trade_parameters(direction,100,0,10000,1.5,10)
        return TripleBarrierPosition(plan,book(),funding,hold_seconds=hold,fee=.0004)

    def test_first_hit_not_final_price(self):
        p=self.position()
        hit=p.update(book(99,1))
        self.assertEqual(hit['label'],'LOSS')
        self.assertEqual(p.update(book(105,2)),hit)

    def test_short_exits_at_ask_and_wins(self):
        p=self.position('SHORT')
        r=p.update(book(98,1))
        self.assertEqual(r['label'],'WIN')
        self.assertAlmostEqual(r['exit_price'],98.01)
        self.assertLess(r['funding_return'],0)

    def test_vertical_is_neutral_with_nonzero_return(self):
        p=self.position(hold=5)
        r=p.update(book(100.2,5))
        self.assertEqual(r['label'],'NEUTRAL')
        self.assertNotEqual(r['net_return'],0)
        expected=(r['exit_price']/r['entry_price']-1)-.0004*(1+r['exit_price']/r['entry_price'])-.0008*5/28800
        self.assertAlmostEqual(r['net_return'],expected)

    def test_after_deadline_does_not_create_win(self):
        p=self.position(hold=5)
        self.assertEqual(p.update(book(105,6))['label'],'NEUTRAL')

    def test_gap_and_incomplete_depth(self):
        p=self.position()
        with self.assertRaises(ValueError):
            p.update(book(100,settings.MAX_QUOTE_GAP_SECONDS+1))
        with self.assertRaises(ValueError):
            vwap([[100,1]],2)
        self.assertAlmostEqual(vwap([[100,1],[102,1]],2),101)


class DataTests(unittest.TestCase):
    def test_top20_and_contract_conversion(self):
        raw={'bids':[[99-i,2] for i in range(25)],'asks':[[101+i,6] for i in range(25)]}
        b=normalize_book(raw,1,.01)
        self.assertEqual(len(b['bids']),20)
        self.assertAlmostEqual(b['orderbook_imbalance'],-.5)
        self.assertAlmostEqual(b['bids'][0][1],.02)

    def test_sma200_and_closed_candle(self):
        rows=[[i*900000,100+i,101+i,99+i,100+i,1] for i in range(231)]
        r=indicators(rows,'15m',230*900+10)
        self.assertEqual(r['last_closed_candle_ms'],229*900000)
        self.assertTrue(math.isfinite(r['sma_200']))
        with self.assertRaises(ValueError):
            indicators(rows[:50],'15m',1e9)

    def test_schema_validation(self):
        valid=dict(steps=dict(momentum='trend',liquidity='depth',risk_inv='risk'),direction='LONG',confidence=.5,rationale='evidence')
        Decision.model_validate(valid)
        for changes in [dict(confidence=.99),dict(direction='BUY'),dict(rationale='word '*101)]:
            with self.assertRaises(ValueError):
                Decision.model_validate({**valid,**changes})


class StatisticsTests(unittest.TestCase):
    def test_stationary_bootstrap_and_zero_difference(self):
        r=paired_analysis([0,.01,.02],[0,0,0],resamples=500)
        self.assertAlmostEqual(r['mean_difference'],.01)
        self.assertEqual(r,paired_analysis([0,.01,.02],[0,0,0],resamples=500))
        self.assertEqual(paired_analysis([0,0,0],[0,0,0])['wilcoxon_pvalue'],1)

    def test_agreement_and_holm(self):
        perfect=[[1,1,1],[2,2,2],[5,5,5]]
        self.assertAlmostEqual(ordinal_alpha(perfect),1.)
        self.assertAlmostEqual(fleiss_binary([[0,0,0],[1,1,1]]),1.)
        self.assertIsNone(ordinal_alpha([[1,1,1],[1,1,1]]))
        self.assertTrue(np.allclose(holm([.01,.04,.03]),[.03,.06,.06]))

    def test_incomplete_run_not_imputed(self):
        with tempfile.TemporaryDirectory() as temp:
            store=EventStore(str(Path(temp)/'test.db'))
            store.append('r',None,'manifest',dict(symbols=['BTC'],epochs_per_symbol=3,ablation=False,equity=10000))
            store.append('r','BTC:1','snapshot',{})
            store.append('r','BTC:1','incomplete',dict(tier='council'))
            report=report_run(store,'r')
            self.assertFalse(report['complete'])
            self.assertNotIn('paired_council_vs_single',report)


class CouncilTests(unittest.TestCase):
    def test_four_calls_dependencies_and_usage(self):
        from council.engine import CouncilEngine
        import json
        calls=[]
        valid=dict(steps=dict(momentum='trend',liquidity='depth',risk_inv='risk'),direction='NO-TRADE',confidence=.5,rationale='invalidated')
        def fake(prompt, instruction='', schema=None):
            calls.append((prompt,instruction))
            return dict(text=json.dumps(valid) if schema else 'evidence',tokens_used=10,api_calls=1,model_used='fake',usage={})
        with patch('council.engine.gemini_router.generate',side_effect=fake):
            r=CouncilEngine().evaluate('BTC',{})
        self.assertEqual(len(calls),4)
        self.assertIn('Desk views:',calls[2][0])
        self.assertIn('CRO:',calls[3][0])
        self.assertEqual(r['tokens_used'],40)


class AuditWorkflowTests(unittest.TestCase):
    def test_packet_roundtrip_requires_real_completed_ratings(self):
        import json
        from benchmark.audit import export_packet, score_packet
        with tempfile.TemporaryDirectory() as temp:
            store=EventStore(str(Path(temp)/'audit.db'))
            decision=dict(steps=dict(momentum='trend',liquidity='depth',risk_inv='risk'),
                          direction='NO-TRADE',confidence=.5,rationale='evidence')
            for i in range(20):
                epoch=str(i)
                store.append('r',epoch,'snapshot',dict(current_price_usd=100))
                for tier in ('single','council'):
                    store.append('r',epoch,'decision',dict(tier=tier,**decision))
            output=Path(temp)/'packet'
            export_packet(store,'r',output)
            with self.assertRaises(ValueError):
                score_packet(output)
            mapping=json.loads((output/'SECRET_AUDIT_KEY.json').read_text())['mapping']
            for i in range(1,4):
                p=output/f'evaluator_{i}.json'
                judge=json.loads(p.read_text())
                judge.update(model=f'fake-test-model-{i}',evaluated_at='2026-11-11T00:00:00Z')
                for case in mapping:
                    for label in ('System_A','System_B'):
                        judge['scores'][case][label]=[5 if mapping[case][label]=='council' else 2]*4
                p.write_text(json.dumps(judge))
            result=score_packet(output)
            self.assertEqual(result['scenarios'],20)
            self.assertEqual(result['krippendorff_ordinal_alpha'],1.)
            self.assertLess(result['dimensions']['market_structure']['holm_p'],.001)
            with self.assertRaises(ValueError):
                export_packet(store,'r',output)


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_quote_stream_directional_fills_and_first_hit_persisted(self):
        from benchmark.runner import QuoteFeed, run_epoch
        class FakeExchange:
            counter=0
            def market(self,symbol):
                return dict(linear=True,contractSize=1)
            async def watch_order_book(self,symbol):
                await asyncio.sleep(.003)
                self.counter+=1
                b=book(100 if self.counter<20 else 98)
                return dict(bids=b['bids'],asks=b['asks'])
        valid=dict(direction='LONG',tokens_used=10,api_calls=1,rationale='test')
        state={'timeframe_analysis':{'15m_execution':{'atr':0}},'funding_rate_8h':.0001}
        with tempfile.TemporaryDirectory() as temp:
            store=EventStore(str(Path(temp)/'fills.db'))
            exchange=FakeExchange()
            feed=QuoteFeed(exchange,'BTC',store,'r')
            stream=asyncio.create_task(feed.run())
            try:
                await feed.next()
                with patch('benchmark.runner.snapshot',return_value=state), \
                     patch('benchmark.runner.baseline_quant.evaluate',side_effect=lambda s:valid.copy()), \
                     patch('benchmark.runner.super_agent.evaluate_market',side_effect=lambda symbol,s:valid.copy()), \
                     patch('benchmark.runner.CouncilEngine.evaluate',side_effect=lambda symbol,s:valid.copy()):
                    equities={k:10000 for k in ('rule','single','council')}
                    await asyncio.wait_for(run_epoch(exchange,'BTC',feed,store,'r','e',equities,{}),5)
                results=store.read('r','outcome')
                self.assertEqual(len(results),3)
                self.assertTrue(all(r['label']=='LOSS' for r in results))
                self.assertEqual(len(store.read('r','entry')),3)
                self.assertEqual(store.read('r','incomplete'),[])
                self.assertTrue(all(e<10000 for e in equities.values()))
            finally:
                stream.cancel()
                await asyncio.gather(stream,return_exceptions=True)

    async def test_shared_snapshot_concurrent_tiers_and_valid_cash_outcomes(self):
        import time
        from benchmark.runner import run_epoch
        contexts=[]
        start_times=[]
        end_times=[]
        import threading
        barrier=threading.Barrier(3)
        state={'timeframe_analysis':{'15m_execution':{'atr':1}},'funding_rate_8h':.0001}
        class FakeFeed:
            async def next(self,after=0):
                return book(now=time.time()+.001)
        def evaluate(*args):
            context=args[-1]
            contexts.append(copy.deepcopy(context))
            start_times.append(time.perf_counter())
            context['mutation']='must stay isolated'
            barrier.wait(timeout=5)
            end_times.append(time.perf_counter())
            return dict(direction='NO-TRADE',tokens_used=0,api_calls=0,rationale='cash')
        with tempfile.TemporaryDirectory() as temp:
            store=EventStore(str(Path(temp)/'runner.db'))
            with patch('benchmark.runner.snapshot',return_value=copy.deepcopy(state)), \
                 patch('benchmark.runner.baseline_quant.evaluate',side_effect=evaluate), \
                 patch('benchmark.runner.super_agent.evaluate_market',side_effect=evaluate), \
                 patch('benchmark.runner.CouncilEngine.evaluate',side_effect=evaluate):
                equities={k:10000 for k in ('rule','single','council')}
                await run_epoch(None,'BTC',FakeFeed(),store,'r','e',equities,{})
            self.assertEqual(len(contexts),3)
            self.assertEqual(contexts[0],contexts[1])
            self.assertEqual(contexts[1],contexts[2])
            self.assertLess(max(start_times),min(end_times))
            outcomes=store.read('r','outcome')
            self.assertEqual(len(outcomes),3)
            self.assertTrue(all(r['label']=='NO-TRADE' for r in outcomes))
            self.assertEqual(store.read('r','incomplete'),[])

    async def test_model_failure_remains_incomplete(self):
        import time
        from benchmark.runner import run_epoch
        class FakeFeed:
            async def next(self,after=0):
                return book(now=time.time()+.001)
        valid=dict(direction='NO-TRADE',tokens_used=0,api_calls=0)
        with tempfile.TemporaryDirectory() as temp:
            store=EventStore(str(Path(temp)/'failure.db'))
            with patch('benchmark.runner.snapshot',return_value={}), \
                 patch('benchmark.runner.baseline_quant.evaluate',return_value=valid.copy()), \
                 patch('benchmark.runner.super_agent.evaluate_market',side_effect=ValueError('invalid schema')), \
                 patch('benchmark.runner.CouncilEngine.evaluate',return_value=valid.copy()):
                await run_epoch(None,'BTC',FakeFeed(),store,'r','e',
                                {k:10000 for k in ('rule','single','council')},{})
            self.assertEqual(len(store.read('r','outcome')),2)
            self.assertEqual(store.read('r','incomplete')[0]['tier'],'single')

    async def test_binance_funding_period_adjustments(self):
        from tools.telemetry import funding_observation
        class FakeExchange:
            id='binanceusdm'
            async def fetch_funding_rate(self,symbol):
                return dict(fundingRate=.0002,interval=None)
            async def fetch_funding_intervals(self,symbols):
                return {'BTC':{'interval':'4h'}}
        self.assertEqual((await funding_observation(FakeExchange(),'BTC'))['interval'],'4h')
        self.assertEqual((await funding_observation(FakeExchange(),'ETH'))['interval'],'8h')


if __name__=='__main__':
    unittest.main()
