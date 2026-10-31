-- =====================================================================================
-- 37 SQL BUSINESS QUESTIONS  (PostgreSQL 16)  -- all tested by src/database/run_queries.py
-- Conventions used throughout (stated once, applied everywhere):
--   * AS-OF DATE = 2026-06-30 (last day of the viewing data). Never CURRENT_DATE.
--   * "Active customer"   = churn_labels.churned = FALSE
--   * "Current plan/sub"  = the customer's latest subscriptions row (by start date)
--   * "Revenue/spend"     = payments with payment_status = 'Success' (cleaned amount)
--   * Payments data ends 2026-05-31, so June 2026 has no billing rows.
-- Each query starts with a "-- QNN" marker line; the runner splits on it.
-- =====================================================================================

-- ############################ SECTION A: INTERMEDIATE ############################

-- Q01 | Active customers by country
SELECT c.country, COUNT(*) AS active_customers
FROM customers c JOIN churn_labels cl USING (customer_id)
WHERE cl.churned = FALSE
GROUP BY c.country ORDER BY active_customers DESC;

-- Q02 | Total and average monthly revenue per plan (successful payments, linked via subscription)
WITH monthly AS (
  SELECT sp.plan_name, DATE_TRUNC('month', p.payment_date) AS month, SUM(p.amount) AS revenue
  FROM payments p
  JOIN subscriptions s ON s.subscription_id = p.subscription_id
  JOIN subscription_plans sp ON sp.plan_id = s.plan_id
  WHERE p.payment_status = 'Success'
  GROUP BY sp.plan_name, DATE_TRUNC('month', p.payment_date)
)
SELECT plan_name, ROUND(SUM(revenue),2) AS total_revenue,
       ROUND(AVG(revenue),2) AS avg_monthly_revenue, COUNT(*) AS months_with_billing
FROM monthly GROUP BY plan_name ORDER BY total_revenue DESC;

-- Q03 | Overall churn rate
SELECT COUNT(*) AS customers, SUM(churned::int) AS churned_customers,
       ROUND(100.0 * SUM(churned::int) / COUNT(*), 2) AS churn_rate_pct
FROM churn_labels;

-- Q04 | Customers per acquisition channel + churn rate
SELECT c.acquisition_channel, COUNT(*) AS customers,
       ROUND(100.0 * AVG(cl.churned::int), 2) AS churn_rate_pct
FROM customers c JOIN churn_labels cl USING (customer_id)
GROUP BY c.acquisition_channel ORDER BY customers DESC;

-- Q05 | Average customer age by (current) plan
WITH cur AS (
  SELECT DISTINCT ON (customer_id) customer_id, plan_id
  FROM subscriptions ORDER BY customer_id, subscription_start_date DESC
)
SELECT sp.plan_name, ROUND(AVG(c.age),1) AS avg_age, COUNT(*) AS customers
FROM cur JOIN customers c USING (customer_id) JOIN subscription_plans sp USING (plan_id)
GROUP BY sp.plan_name ORDER BY avg_age;

-- Q06 | Top 10 most-watched titles (total minutes)
SELECT ct.content_id, ct.title, ct.content_type, ct.genre, SUM(v.watch_duration_minutes) AS total_minutes
FROM viewing_activity v JOIN content ct USING (content_id)
GROUP BY ct.content_id, ct.title, ct.content_type, ct.genre
ORDER BY total_minutes DESC LIMIT 10;

-- Q07 | Average completion % by content_type and genre
SELECT ct.content_type, ct.genre, ROUND(AVG(v.completion_percentage),2) AS avg_completion_pct, COUNT(*) AS sessions
FROM viewing_activity v JOIN content ct USING (content_id)
GROUP BY ct.content_type, ct.genre ORDER BY ct.content_type, avg_completion_pct DESC;

-- Q08 | Failed payments in each of the last 12 months (zero-filled; billing data ends May 2026)
WITH months AS (
  SELECT generate_series(DATE '2025-07-01', DATE '2026-06-01', INTERVAL '1 month')::date AS month
)
SELECT m.month, COUNT(p.payment_id) AS failed_payments
FROM months m
LEFT JOIN payments p ON DATE_TRUNC('month', p.payment_date) = m.month AND p.payment_status = 'Failed'
GROUP BY m.month ORDER BY m.month;

-- Q09 | Average resolution time by issue category (resolved/closed tickets only)
SELECT issue_category, ROUND(AVG(resolution_time_hours),1) AS avg_resolution_hours, COUNT(*) AS tickets
FROM support_tickets WHERE resolution_time_hours IS NOT NULL
GROUP BY issue_category ORDER BY avg_resolution_hours DESC;

-- Q10 | Distribution of CSAT across resolved/closed tickets
SELECT customer_satisfaction_score AS csat, COUNT(*) AS tickets,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
FROM support_tickets
WHERE ticket_status IN ('Resolved','Closed') AND customer_satisfaction_score IS NOT NULL
GROUP BY customer_satisfaction_score ORDER BY csat;

-- Q11 | Device generating the most viewing minutes
SELECT device_type, SUM(watch_duration_minutes) AS total_minutes,
       ROUND(100.0 * SUM(watch_duration_minutes) / SUM(SUM(watch_duration_minutes)) OVER (), 1) AS pct_of_total
FROM viewing_activity GROUP BY device_type ORDER BY total_minutes DESC;

-- Q12 | Average rating by sentiment label (shows the label noise: Positive should not average ~3)
SELECT sentiment_label, ROUND(AVG(rating),2) AS avg_rating, COUNT(*) AS responses
FROM customer_feedback GROUP BY sentiment_label ORDER BY avg_rating DESC;

-- Q13 | Signups by acquisition channel per calendar year (pivoted)
SELECT acquisition_channel,
  COUNT(*) FILTER (WHERE EXTRACT(YEAR FROM registration_date)=2019) AS y2019,
  COUNT(*) FILTER (WHERE EXTRACT(YEAR FROM registration_date)=2020) AS y2020,
  COUNT(*) FILTER (WHERE EXTRACT(YEAR FROM registration_date)=2021) AS y2021,
  COUNT(*) FILTER (WHERE EXTRACT(YEAR FROM registration_date)=2022) AS y2022,
  COUNT(*) FILTER (WHERE EXTRACT(YEAR FROM registration_date)=2023) AS y2023,
  COUNT(*) FILTER (WHERE EXTRACT(YEAR FROM registration_date)=2024) AS y2024,
  COUNT(*) FILTER (WHERE EXTRACT(YEAR FROM registration_date)=2025) AS y2025,
  COUNT(*) FILTER (WHERE EXTRACT(YEAR FROM registration_date)=2026) AS y2026,
  COUNT(*) AS total
FROM customers GROUP BY acquisition_channel ORDER BY total DESC;

-- Q14 | % of customers with auto-renew OFF (on their current subscription)
WITH cur AS (
  SELECT DISTINCT ON (customer_id) customer_id, auto_renew
  FROM subscriptions ORDER BY customer_id, subscription_start_date DESC
)
SELECT COUNT(*) AS customers, SUM((NOT auto_renew)::int) AS auto_renew_off,
       ROUND(100.0 * SUM((NOT auto_renew)::int) / COUNT(*), 2) AS pct_auto_renew_off
FROM cur;

-- Q15 | Customers with more than 3 support tickets
SELECT c.customer_id, c.first_name, c.last_name, c.country, COUNT(*) AS ticket_count
FROM support_tickets t JOIN customers c USING (customer_id)
GROUP BY c.customer_id, c.first_name, c.last_name, c.country
HAVING COUNT(*) > 3 ORDER BY ticket_count DESC, c.customer_id;

-- ############################ SECTION B: ADVANCED ############################

-- Q16 | RANK() plans by revenue = monthly_price x count of ACTIVE subscriptions
SELECT sp.plan_name, COUNT(*) AS active_subscriptions,
       ROUND(SUM(s.monthly_price),2) AS monthly_run_rate,
       RANK() OVER (ORDER BY SUM(s.monthly_price) DESC) AS revenue_rank
FROM subscriptions s JOIN subscription_plans sp USING (plan_id)
WHERE s.subscription_status = 'Active'
GROUP BY sp.plan_name ORDER BY revenue_rank;

-- Q17 | Most recent support ticket per customer (ROW_NUMBER); invalid dates excluded
WITH ranked AS (
  SELECT t.*, ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY ticket_date DESC, ticket_id DESC) AS rn
  FROM support_tickets t WHERE ticket_date IS NOT NULL
)
SELECT customer_id, ticket_id, ticket_date, issue_category, issue_subcategory, priority, ticket_status,
       customer_satisfaction_score
FROM ranked WHERE rn = 1 ORDER BY customer_id;

-- Q18 | 3-month moving average of new registrations
WITH monthly AS (
  SELECT m::date AS month, COUNT(c.customer_id) AS registrations
  FROM generate_series((SELECT DATE_TRUNC('month', MIN(registration_date)) FROM customers),
                       (SELECT DATE_TRUNC('month', MAX(registration_date)) FROM customers), INTERVAL '1 month') m
  LEFT JOIN customers c ON DATE_TRUNC('month', c.registration_date) = m
  GROUP BY m
)
SELECT month, registrations,
       ROUND(AVG(registrations) OVER (ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW),2) AS moving_avg_3m
FROM monthly ORDER BY month;

-- Q19 | LAG(): customers whose watch minutes dropped in the latest month (Jun-26) vs the prior month (May-26)
WITH monthly AS (
  SELECT customer_id, DATE_TRUNC('month', viewing_date)::date AS month, SUM(watch_duration_minutes) AS minutes
  FROM viewing_activity GROUP BY 1, 2
), grid AS (  -- zero-fill so "stopped watching" counts as a decline
  SELECT c.customer_id, m::date AS month, COALESCE(mo.minutes, 0) AS minutes
  FROM (SELECT DISTINCT customer_id FROM viewing_activity) c
  CROSS JOIN generate_series(DATE '2025-07-01', DATE '2026-06-01', INTERVAL '1 month') m
  LEFT JOIN monthly mo ON mo.customer_id = c.customer_id AND mo.month = m::date
), lagged AS (
  SELECT *, LAG(minutes) OVER (PARTITION BY customer_id ORDER BY month) AS prev_minutes FROM grid
)
SELECT customer_id, prev_minutes AS may_minutes, minutes AS june_minutes,
       ROUND(100.0 * (minutes - prev_minutes) / NULLIF(prev_minutes,0), 1) AS pct_change
FROM lagged
WHERE month = DATE '2026-06-01' AND prev_minutes > 0 AND minutes < prev_minutes
ORDER BY pct_change;

-- Q20 | Registration-cohort retention at 1, 3, 6 months (only cohorts old enough to be observed)
-- NOTE: churn dates only span Oct-2025..Jun-2026, so older cohorts look artificially retained early on.
WITH cust AS (
  SELECT c.customer_id, DATE_TRUNC('month', c.registration_date)::date AS cohort, c.registration_date,
         cl.churn_date
  FROM customers c JOIN churn_labels cl USING (customer_id)
)
SELECT cohort, COUNT(*) AS cohort_size,
  ROUND(100.0 * COUNT(*) FILTER (WHERE registration_date + INTERVAL '1 month' <= DATE '2026-06-30'
        AND (churn_date IS NULL OR churn_date > registration_date + INTERVAL '1 month'))
        / NULLIF(COUNT(*) FILTER (WHERE registration_date + INTERVAL '1 month' <= DATE '2026-06-30'),0), 1) AS retained_1m_pct,
  ROUND(100.0 * COUNT(*) FILTER (WHERE registration_date + INTERVAL '3 months' <= DATE '2026-06-30'
        AND (churn_date IS NULL OR churn_date > registration_date + INTERVAL '3 months'))
        / NULLIF(COUNT(*) FILTER (WHERE registration_date + INTERVAL '3 months' <= DATE '2026-06-30'),0), 1) AS retained_3m_pct,
  ROUND(100.0 * COUNT(*) FILTER (WHERE registration_date + INTERVAL '6 months' <= DATE '2026-06-30'
        AND (churn_date IS NULL OR churn_date > registration_date + INTERVAL '6 months'))
        / NULLIF(COUNT(*) FILTER (WHERE registration_date + INTERVAL '6 months' <= DATE '2026-06-30'),0), 1) AS retained_6m_pct
FROM cust GROUP BY cohort ORDER BY cohort;

-- Q21 | DENSE_RANK(): top 5 countries by churn rate
WITH rates AS (
  SELECT c.country, COUNT(*) AS customers, ROUND(100.0 * AVG(cl.churned::int),2) AS churn_rate_pct
  FROM customers c JOIN churn_labels cl USING (customer_id)
  GROUP BY c.country HAVING COUNT(*) >= 30
), ranked AS (SELECT *, DENSE_RANK() OVER (ORDER BY churn_rate_pct DESC) AS churn_rank FROM rates)
SELECT * FROM ranked WHERE churn_rank <= 5 ORDER BY churn_rank, country;

-- Q22 | Running total of successful payments per customer
SELECT customer_id, payment_id, payment_date, amount,
       SUM(amount) OVER (PARTITION BY customer_id ORDER BY payment_date, payment_id
                         ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_total
FROM payments WHERE payment_status = 'Success'
ORDER BY customer_id, payment_date, payment_id;

-- Q23 | Customers with 3 consecutive monthly declines in watch time (m < m-1 < m-2); latest such window returned
WITH monthly AS (
  SELECT customer_id, DATE_TRUNC('month', viewing_date)::date AS month, SUM(watch_duration_minutes) AS minutes
  FROM viewing_activity GROUP BY 1, 2
), grid AS (
  SELECT c.customer_id, m::date AS month, COALESCE(mo.minutes, 0) AS minutes
  FROM (SELECT DISTINCT customer_id FROM viewing_activity) c
  CROSS JOIN generate_series(DATE '2025-07-01', DATE '2026-06-01', INTERVAL '1 month') m
  LEFT JOIN monthly mo ON mo.customer_id = c.customer_id AND mo.month = m::date
), lagged AS (
  SELECT *, LAG(minutes,1) OVER w AS m1, LAG(minutes,2) OVER w AS m2
  FROM grid WINDOW w AS (PARTITION BY customer_id ORDER BY month)
), hits AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY month DESC) AS rn
  FROM lagged WHERE minutes < m1 AND m1 < m2
)
SELECT customer_id, month AS window_end_month, m2 AS minutes_m_minus_2, m1 AS minutes_m_minus_1, minutes AS minutes_m
FROM hits WHERE rn = 1 ORDER BY month DESC, customer_id;

-- Q24 | Monthly churn rate, last 12 months = churned in month / customers active at start of month
WITH months AS (SELECT generate_series(DATE '2025-07-01', DATE '2026-06-01', INTERVAL '1 month')::date AS month)
SELECT m.month,
  COUNT(*) FILTER (WHERE c.registration_date < m.month AND (cl.churn_date IS NULL OR cl.churn_date >= m.month)) AS active_at_start,
  COUNT(*) FILTER (WHERE cl.churned AND DATE_TRUNC('month', cl.churn_date) = m.month) AS churned_in_month,
  ROUND(100.0 * COUNT(*) FILTER (WHERE cl.churned AND DATE_TRUNC('month', cl.churn_date) = m.month)
        / NULLIF(COUNT(*) FILTER (WHERE c.registration_date < m.month AND (cl.churn_date IS NULL OR cl.churn_date >= m.month)),0), 2) AS monthly_churn_rate_pct
FROM months m CROSS JOIN customers c JOIN churn_labels cl USING (customer_id)
GROUP BY m.month ORDER BY m.month;

-- Q25 | Retention rate (still active today) by acquisition channel
SELECT c.acquisition_channel, COUNT(*) AS customers,
       ROUND(100.0 * AVG((NOT cl.churned)::int), 2) AS retention_rate_pct
FROM customers c JOIN churn_labels cl USING (customer_id)
GROUP BY c.acquisition_channel ORDER BY retention_rate_pct DESC;

-- Q26 | Lifetime spend quartiles with NTILE(4) (1 = top spenders)
WITH spend AS (
  SELECT c.customer_id, COALESCE(SUM(p.amount) FILTER (WHERE p.payment_status='Success'),0) AS lifetime_spend
  FROM customers c LEFT JOIN payments p USING (customer_id) GROUP BY c.customer_id
), q AS (SELECT *, NTILE(4) OVER (ORDER BY lifetime_spend DESC) AS spend_quartile FROM spend)
SELECT spend_quartile, COUNT(*) AS customers, ROUND(MIN(lifetime_spend),2) AS min_spend,
       ROUND(MAX(lifetime_spend),2) AS max_spend, ROUND(AVG(lifetime_spend),2) AS avg_spend
FROM q GROUP BY spend_quartile ORDER BY spend_quartile;

-- Q27 | Top 3 issue_subcategory per country by ticket volume
WITH v AS (
  SELECT c.country, t.issue_subcategory, COUNT(*) AS tickets
  FROM support_tickets t JOIN customers c USING (customer_id) GROUP BY 1, 2
), r AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY country ORDER BY tickets DESC, issue_subcategory) AS rn FROM v)
SELECT country, rn AS rank, issue_subcategory, tickets FROM r WHERE rn <= 3 ORDER BY country, rn;

-- ############################ SECTION C: EXPERT ############################
-- Shared risk logic (used by Q28, Q31, Q37): composite risk = mean of three min-max normalised signals
--   recency (days since last viewing), failed-payment rate, support ticket count. Higher = riskier.

-- Q28 | 2+ failed payments in last 90 days AND high churn risk (top-quartile composite risk among active customers)
WITH asof AS (SELECT DATE '2026-06-30' AS d),
cur AS (SELECT DISTINCT ON (customer_id) customer_id, plan_id FROM subscriptions ORDER BY customer_id, subscription_start_date DESC),
act AS (SELECT customer_id FROM churn_labels WHERE NOT churned),
rec AS (SELECT customer_id, (SELECT d FROM asof) - MAX(viewing_date) AS days_since FROM viewing_activity GROUP BY 1),
fp AS (SELECT customer_id, AVG((payment_status='Failed')::int) AS fail_rate FROM payments GROUP BY 1),
tk AS (SELECT customer_id, COUNT(*) AS tickets FROM support_tickets GROUP BY 1),
base AS (
  SELECT a.customer_id, COALESCE(r.days_since, 365) AS days_since, COALESCE(f.fail_rate,0) AS fail_rate, COALESCE(t.tickets,0) AS tickets
  FROM act a LEFT JOIN rec r USING (customer_id) LEFT JOIN fp f USING (customer_id) LEFT JOIN tk t USING (customer_id)
), scored AS (
  SELECT customer_id,
    ( (days_since - MIN(days_since) OVER ()) / NULLIF(MAX(days_since) OVER () - MIN(days_since) OVER (),0)
    + (fail_rate - MIN(fail_rate) OVER ()) / NULLIF(MAX(fail_rate) OVER () - MIN(fail_rate) OVER (),0)
    + (tickets - MIN(tickets) OVER ()) / NULLIF(MAX(tickets) OVER () - MIN(tickets) OVER (),0) ) / 3.0 AS risk
  FROM base
), flagged AS (SELECT *, NTILE(4) OVER (ORDER BY risk DESC) AS risk_quartile FROM scored),
recent_fail AS (
  SELECT customer_id, COUNT(*) AS failed_payment_count FROM payments
  WHERE payment_status='Failed' AND payment_date > (SELECT d FROM asof) - 90 GROUP BY 1 HAVING COUNT(*) >= 2
)
SELECT rf.customer_id, rf.failed_payment_count, sp.plan_name AS current_plan, ROUND(f.risk::numeric,3) AS composite_risk
FROM recent_fail rf JOIN flagged f USING (customer_id) JOIN cur USING (customer_id)
JOIN subscription_plans sp ON sp.plan_id = cur.plan_id
WHERE f.risk_quartile = 1
ORDER BY rf.failed_payment_count DESC, f.risk DESC;

-- Q29 | Engagement 30d before vs 30d after each customer's most recent PRICE CHANGE
-- DATA FINDING: subscriptions.monthly_price never differs between a customer's consecutive rows (it is not
-- the price at the time of each segment, despite the dictionary). The only real price change is a PLAN change,
-- so a price change = consecutive subscription rows whose plan LIST price (subscription_plans) differs.
-- Only changes whose full +/-30d windows lie inside the viewing data (2025-07-01..2026-06-30) are measurable.
WITH seq AS (
  SELECT s.customer_id, s.subscription_start_date AS change_date, sp.monthly_price AS new_price,
         LAG(sp.monthly_price) OVER (PARTITION BY s.customer_id ORDER BY s.subscription_start_date) AS prev_price
  FROM subscriptions s JOIN subscription_plans sp USING (plan_id)
), chg AS (
  SELECT customer_id, change_date, prev_price, new_price,
         ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY change_date DESC) AS rn
  FROM seq WHERE prev_price IS NOT NULL AND prev_price <> new_price
), latest AS (
  SELECT * FROM chg WHERE rn = 1 AND change_date BETWEEN DATE '2025-07-31' AND DATE '2026-05-31'
), eng AS (
  SELECT l.customer_id, l.change_date, l.prev_price, l.new_price,
    AVG(v.watch_duration_minutes) FILTER (WHERE v.viewing_date >= l.change_date - 30 AND v.viewing_date < l.change_date) AS before_avg,
    AVG(v.watch_duration_minutes) FILTER (WHERE v.viewing_date >= l.change_date AND v.viewing_date < l.change_date + 30) AS after_avg
  FROM latest l LEFT JOIN viewing_activity v USING (customer_id) GROUP BY 1,2,3,4
)
SELECT customer_id, change_date, prev_price, new_price,
       ROUND(before_avg,1) AS avg_min_before, ROUND(after_avg,1) AS avg_min_after,
       ROUND(100.0*(after_avg-before_avg)/NULLIF(before_avg,0),1) AS pct_change
FROM eng WHERE before_avg IS NOT NULL AND after_avg IS NOT NULL ORDER BY change_date;

-- Q30 | Country scorecard: customers, churned, churn rate, avg CLV of churned customers, top churn reason
WITH base AS (
  SELECT c.customer_id, c.country, cl.churned, cl.churn_reason FROM customers c JOIN churn_labels cl USING (customer_id)
), clv AS (
  SELECT customer_id, SUM(amount) AS clv FROM payments WHERE payment_status='Success' GROUP BY 1
), reasons AS (
  SELECT country, churn_reason, COUNT(*) AS n, ROW_NUMBER() OVER (PARTITION BY country ORDER BY COUNT(*) DESC, churn_reason) AS rn
  FROM base WHERE churned GROUP BY country, churn_reason
), agg AS (
  SELECT b.country, COUNT(*) AS total_customers, COUNT(*) FILTER (WHERE b.churned) AS churned_customers,
         AVG(c.clv) FILTER (WHERE b.churned) AS avg_clv_churned
  FROM base b LEFT JOIN clv c USING (customer_id) GROUP BY b.country
)
SELECT a.country, a.total_customers, a.churned_customers,
       ROUND(100.0*a.churned_customers/a.total_customers,2) AS churn_rate_pct,
       ROUND(a.avg_clv_churned,2) AS avg_clv_of_churned, r.churn_reason AS top_churn_reason
FROM agg a LEFT JOIN reasons r ON r.country = a.country AND r.rn = 1
ORDER BY churn_rate_pct DESC;

-- Q31 | At-risk revenue: composite risk (0-1) x monthly price, top 50 ACTIVE customers
WITH asof AS (SELECT DATE '2026-06-30' AS d),
cur AS (SELECT DISTINCT ON (customer_id) customer_id, plan_id, monthly_price FROM subscriptions ORDER BY customer_id, subscription_start_date DESC),
act AS (SELECT customer_id FROM churn_labels WHERE NOT churned),
rec AS (SELECT customer_id, (SELECT d FROM asof) - MAX(viewing_date) AS days_since FROM viewing_activity GROUP BY 1),
fp AS (SELECT customer_id, AVG((payment_status='Failed')::int) AS fail_rate FROM payments GROUP BY 1),
tk AS (SELECT customer_id, COUNT(*) AS tickets FROM support_tickets GROUP BY 1),
base AS (
  SELECT a.customer_id, COALESCE(r.days_since, 365) AS days_since, COALESCE(f.fail_rate,0) AS fail_rate, COALESCE(t.tickets,0) AS tickets
  FROM act a LEFT JOIN rec r USING (customer_id) LEFT JOIN fp f USING (customer_id) LEFT JOIN tk t USING (customer_id)
), scored AS (
  SELECT *, ( (days_since - MIN(days_since) OVER ()) / NULLIF(MAX(days_since) OVER () - MIN(days_since) OVER (),0)
    + (fail_rate - MIN(fail_rate) OVER ()) / NULLIF(MAX(fail_rate) OVER () - MIN(fail_rate) OVER (),0)
    + (tickets - MIN(tickets) OVER ()) / NULLIF(MAX(tickets) OVER () - MIN(tickets) OVER (),0) ) / 3.0 AS risk
  FROM base
)
SELECT s.customer_id, sp.plan_name, cur.monthly_price, s.days_since AS days_since_last_activity,
       ROUND(s.fail_rate::numeric,3) AS failed_payment_rate, s.tickets AS support_ticket_count,
       ROUND(s.risk::numeric,3) AS composite_risk, ROUND((s.risk * cur.monthly_price)::numeric,2) AS at_risk_revenue,
       RANK() OVER (ORDER BY s.risk * cur.monthly_price DESC) AS rank
FROM scored s JOIN cur USING (customer_id) JOIN subscription_plans sp ON sp.plan_id = cur.plan_id
ORDER BY at_risk_revenue DESC LIMIT 50;

-- Q32 | Top 5 complaint subcategories per country, most recent completed quarter (Q2 2026), with avg resolution & CSAT
WITH q AS (
  SELECT (DATE_TRUNC('quarter', DATE '2026-06-30' + 1) - INTERVAL '3 months')::date AS q_start,
         DATE_TRUNC('quarter', DATE '2026-06-30' + 1)::date AS q_end
), t AS (
  SELECT c.country, tk.issue_subcategory, COUNT(*) AS tickets,
         ROUND(AVG(tk.resolution_time_hours),1) AS avg_resolution_hours,
         ROUND(AVG(tk.customer_satisfaction_score),2) AS avg_csat
  FROM support_tickets tk JOIN customers c USING (customer_id), q
  WHERE tk.ticket_date >= q.q_start AND tk.ticket_date < q.q_end GROUP BY c.country, tk.issue_subcategory
), r AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY country ORDER BY tickets DESC, issue_subcategory) AS rn FROM t)
SELECT country, rn AS rank, issue_subcategory, tickets, avg_resolution_hours, avg_csat
FROM r WHERE rn <= 5 ORDER BY country, rn;

-- Q33 | Win-back candidates: churned customers whose PRE-churn lifetime value is in the top quartile (all customers ranked)
WITH pre AS (
  SELECT cl.customer_id, cl.churn_date, cl.churn_reason,
         COALESCE(SUM(p.amount) FILTER (WHERE p.payment_status='Success' AND p.payment_date <= cl.churn_date),0) AS clv_before_churn
  FROM churn_labels cl LEFT JOIN payments p USING (customer_id) WHERE cl.churned GROUP BY 1,2,3
), allclv AS (
  SELECT customer_id, SUM(amount) AS clv FROM payments WHERE payment_status='Success' GROUP BY 1
), q AS (
  SELECT pre.*, NTILE(4) OVER (ORDER BY clv_before_churn DESC) AS clv_quartile FROM pre
), eng AS (
  SELECT q.customer_id,
    COUNT(v.viewing_id) FILTER (WHERE v.viewing_date BETWEEN q.churn_date - 30 AND q.churn_date) AS sessions_last_30d,
    ROUND(AVG(v.watch_duration_minutes) FILTER (WHERE v.viewing_date BETWEEN q.churn_date - 30 AND q.churn_date),1) AS avg_watch_last_30d
  FROM q LEFT JOIN viewing_activity v USING (customer_id) GROUP BY q.customer_id
)
SELECT q.customer_id, q.churn_date, q.churn_reason, ROUND(q.clv_before_churn,2) AS clv_before_churn,
       e.sessions_last_30d, e.avg_watch_last_30d,
       CASE WHEN e.sessions_last_30d >= 4 THEN 'High' WHEN e.sessions_last_30d >= 1 THEN 'Medium' ELSE 'Low' END AS last_known_engagement
FROM q JOIN eng e USING (customer_id) WHERE q.clv_quartile = 1 ORDER BY q.clv_before_churn DESC;
-- (NOTE: NTILE here ranks churned customers only; "top quartile of churned" is the practical win-back pool.)

-- Q34 | Plan migration: most common paths and churn of upgraders vs downgraders
WITH seq AS (
  SELECT s.customer_id, sp.plan_name, sp.monthly_price AS list_price, s.subscription_start_date,
         LAG(sp.plan_name) OVER w AS from_plan, LAG(sp.monthly_price) OVER w AS from_price
  FROM subscriptions s JOIN subscription_plans sp USING (plan_id)
  WINDOW w AS (PARTITION BY s.customer_id ORDER BY s.subscription_start_date)
), moves AS (
  SELECT customer_id, from_plan, plan_name AS to_plan,
         CASE WHEN list_price > from_price THEN 'Upgrade' WHEN list_price < from_price THEN 'Downgrade' ELSE 'Same tier' END AS direction
  FROM seq WHERE from_plan IS NOT NULL
)
SELECT m.from_plan || ' -> ' || m.to_plan AS migration_path, m.direction, COUNT(*) AS customers,
       ROUND(100.0 * AVG(cl.churned::int), 2) AS churn_rate_pct,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct_of_migrations,
       RANK() OVER (ORDER BY COUNT(*) DESC) AS popularity_rank
FROM moves m JOIN churn_labels cl USING (customer_id)
GROUP BY m.from_plan, m.to_plan, m.direction ORDER BY customers DESC;

-- Q35 | WAU / MAU stickiness, last 6 months (WAU = average distinct weekly actives within the month)
WITH wk AS (
  SELECT DATE_TRUNC('week', viewing_date)::date AS week_start, COUNT(DISTINCT customer_id) AS wau
  FROM viewing_activity WHERE viewing_date >= DATE '2026-01-01' GROUP BY 1
), wk_m AS (
  SELECT DATE_TRUNC('month', week_start)::date AS month, ROUND(AVG(wau),0) AS avg_wau, COUNT(*) AS weeks
  FROM wk GROUP BY 1
), mau AS (
  SELECT DATE_TRUNC('month', viewing_date)::date AS month, COUNT(DISTINCT customer_id) AS mau
  FROM viewing_activity WHERE viewing_date >= DATE '2026-01-01' GROUP BY 1
)
SELECT mau.month, mau.mau, wk_m.avg_wau, ROUND(100.0 * wk_m.avg_wau / mau.mau, 1) AS stickiness_pct
FROM mau JOIN wk_m USING (month) ORDER BY mau.month;

-- Q36 | Proactive-outreach proxy: ACTIVE customers with negative feedback (rating <= 2) within 14 days AFTER a failed payment
-- or a low-CSAT (<=2) ticket. One row per customer (most recent qualifying feedback).
WITH neg AS (
  SELECT feedback_id, customer_id, feedback_date, rating FROM customer_feedback WHERE rating <= 2
), triggers AS (
  SELECT customer_id, payment_date AS event_date, 'Failed payment' AS trigger_type FROM payments WHERE payment_status='Failed'
  UNION ALL
  SELECT customer_id, ticket_date, 'Low-CSAT ticket' FROM support_tickets
  WHERE customer_satisfaction_score <= 2 AND ticket_date IS NOT NULL
), hits AS (
  SELECT n.customer_id, n.feedback_id, n.feedback_date, n.rating, t.trigger_type, t.event_date,
         (n.feedback_date - t.event_date) AS days_after_event,
         ROW_NUMBER() OVER (PARTITION BY n.customer_id ORDER BY n.feedback_date DESC, t.event_date DESC) AS rn
  FROM neg n JOIN triggers t ON t.customer_id = n.customer_id AND n.feedback_date BETWEEN t.event_date AND t.event_date + 14
  JOIN churn_labels cl ON cl.customer_id = n.customer_id AND NOT cl.churned
)
SELECT customer_id, feedback_id, feedback_date, rating, trigger_type, event_date, days_after_event
FROM hits WHERE rn = 1 ORDER BY feedback_date DESC, customer_id;

-- Q37 | Customer health score (0-100, 100 = healthiest) for ACTIVE customers, ranked by risk (highest risk first)
-- risk(0-100) = 30% recency + 25% failed-payment rate + 20% low CSAT + 15% plan (Basic highest) + 10% short tenure
WITH asof AS (SELECT DATE '2026-06-30' AS d),
cur AS (SELECT DISTINCT ON (customer_id) customer_id, plan_id FROM subscriptions ORDER BY customer_id, subscription_start_date DESC),
rec AS (SELECT customer_id, (SELECT d FROM asof) - MAX(viewing_date) AS days_since FROM viewing_activity GROUP BY 1),
fp AS (SELECT customer_id, AVG((payment_status='Failed')::int) AS fail_rate FROM payments GROUP BY 1),
cs AS (SELECT customer_id, AVG(customer_satisfaction_score) AS avg_csat FROM support_tickets GROUP BY 1),
base AS (
  SELECT c.customer_id, sp.plan_name, (SELECT d FROM asof) - c.registration_date AS tenure_days,
         COALESCE(r.days_since, 365) AS days_since, COALESCE(f.fail_rate,0) AS fail_rate, cs.avg_csat
  FROM customers c JOIN churn_labels cl USING (customer_id)
  JOIN cur USING (customer_id) JOIN subscription_plans sp ON sp.plan_id = cur.plan_id
  LEFT JOIN rec r USING (customer_id) LEFT JOIN fp f USING (customer_id) LEFT JOIN cs USING (customer_id)
  WHERE NOT cl.churned
), norm AS (
  SELECT *,
    LEAST(days_since,90) / 90.0 AS n_recency,
    LEAST(fail_rate / 0.5, 1.0) AS n_fail,
    COALESCE((5 - avg_csat) / 4.0, 0.5) AS n_csat,            -- no tickets -> neutral 0.5
    CASE plan_name WHEN 'Basic' THEN 1.0 WHEN 'Standard' THEN 0.5 ELSE 0.0 END AS n_plan,
    1 - LEAST(tenure_days, 1825) / 1825.0 AS n_tenure
  FROM base
)
SELECT customer_id, plan_name, tenure_days, days_since AS days_since_last_activity,
       ROUND(fail_rate::numeric,3) AS failed_payment_rate, ROUND(avg_csat::numeric,2) AS avg_csat,
       ROUND((100 * (0.30*n_recency + 0.25*n_fail + 0.20*n_csat + 0.15*n_plan + 0.10*n_tenure))::numeric,1) AS risk_score,
       ROUND((100 - 100 * (0.30*n_recency + 0.25*n_fail + 0.20*n_csat + 0.15*n_plan + 0.10*n_tenure))::numeric,1) AS health_score,
       RANK() OVER (ORDER BY (0.30*n_recency + 0.25*n_fail + 0.20*n_csat + 0.15*n_plan + 0.10*n_tenure) DESC) AS risk_rank
FROM norm ORDER BY risk_rank, customer_id;
