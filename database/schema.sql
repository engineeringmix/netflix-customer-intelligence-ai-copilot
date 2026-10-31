-- Netflix Customer Intelligence Platform: PostgreSQL schema
-- Load order matters (FKs). Run via: python database/load_data.py
-- Additions vs. the schema document (all documented in README): audit/flag columns produced by cleaning.

DROP TABLE IF EXISTS churn_labels, customer_feedback, support_tickets, payments, viewing_activity,
                     content, subscriptions, subscription_plans, customers CASCADE;

CREATE TABLE customers (
  customer_id VARCHAR(12) NOT NULL, first_name VARCHAR(50), last_name VARCHAR(50), age SMALLINT,
  gender VARCHAR(25), country VARCHAR(60), state_or_region VARCHAR(60), city VARCHAR(60),
  registration_date DATE, acquisition_channel VARCHAR(50), customer_segment VARCHAR(30),
  preferred_language VARCHAR(30),
  is_age_outlier BOOLEAN, age_was_missing BOOLEAN,           -- cleaning flags
  CONSTRAINT pk_customers PRIMARY KEY (customer_id)
);

CREATE TABLE subscription_plans (
  plan_id VARCHAR(10) NOT NULL, plan_name VARCHAR(30) NOT NULL, monthly_price NUMERIC(6,2) NOT NULL,
  video_quality VARCHAR(30), max_devices SMALLINT, advertisements VARCHAR(5),
  CONSTRAINT pk_subscription_plans PRIMARY KEY (plan_id)
);

CREATE TABLE subscriptions (
  subscription_id VARCHAR(12) NOT NULL, customer_id VARCHAR(12) NOT NULL, plan_id VARCHAR(10) NOT NULL,
  subscription_start_date DATE NOT NULL, subscription_end_date DATE, subscription_status VARCHAR(20),
  auto_renew BOOLEAN, cancellation_date DATE, cancellation_reason VARCHAR(80), monthly_price NUMERIC(6,2),
  CONSTRAINT pk_subscriptions PRIMARY KEY (subscription_id),
  CONSTRAINT fk_subscriptions_customer FOREIGN KEY (customer_id) REFERENCES customers (customer_id),
  CONSTRAINT fk_subscriptions_plan FOREIGN KEY (plan_id) REFERENCES subscription_plans (plan_id)
);

CREATE TABLE content (
  content_id VARCHAR(10) NOT NULL, title VARCHAR(150) NOT NULL, content_type VARCHAR(30), genre VARCHAR(30),
  release_year SMALLINT, duration_minutes SMALLINT, content_language VARCHAR(30), production_country VARCHAR(60),
  maturity_rating VARCHAR(10),
  CONSTRAINT pk_content PRIMARY KEY (content_id)
);

CREATE TABLE viewing_activity (
  viewing_id VARCHAR(14) NOT NULL, customer_id VARCHAR(12) NOT NULL, content_id VARCHAR(10) NOT NULL,
  viewing_date DATE NOT NULL, watch_duration_minutes SMALLINT, completion_percentage NUMERIC(5,2),
  device_type VARCHAR(30), login_location VARCHAR(60), session_duration_minutes SMALLINT,
  is_watch_outlier BOOLEAN, is_session_outlier BOOLEAN,
  CONSTRAINT pk_viewing_activity PRIMARY KEY (viewing_id),
  CONSTRAINT fk_viewing_customer FOREIGN KEY (customer_id) REFERENCES customers (customer_id),
  CONSTRAINT fk_viewing_content FOREIGN KEY (content_id) REFERENCES content (content_id)
);

CREATE TABLE payments (
  payment_id VARCHAR(14) NOT NULL, customer_id VARCHAR(12) NOT NULL, subscription_id VARCHAR(12),
  payment_date DATE NOT NULL, amount NUMERIC(7,2), payment_method VARCHAR(30), payment_status VARCHAR(15),
  failed_payment_reason VARCHAR(60),
  amount_raw NUMERIC(7,2), is_amount_anomaly BOOLEAN,         -- original value kept for audit
  CONSTRAINT pk_payments PRIMARY KEY (payment_id),
  CONSTRAINT fk_payments_customer FOREIGN KEY (customer_id) REFERENCES customers (customer_id),
  CONSTRAINT fk_payments_subscription FOREIGN KEY (subscription_id) REFERENCES subscriptions (subscription_id)
);

CREATE TABLE support_tickets (
  ticket_id VARCHAR(12) NOT NULL, customer_id VARCHAR(12) NOT NULL, ticket_date DATE,
  issue_category VARCHAR(40), issue_subcategory VARCHAR(60), priority VARCHAR(15), ticket_status VARCHAR(15),
  resolution_time_hours NUMERIC(6,1), customer_satisfaction_score SMALLINT, ticket_description TEXT,
  support_agent_id VARCHAR(10), ticket_date_valid BOOLEAN,
  CONSTRAINT pk_support_tickets PRIMARY KEY (ticket_id),
  CONSTRAINT fk_tickets_customer FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
);

CREATE TABLE customer_feedback (
  feedback_id VARCHAR(12) NOT NULL, customer_id VARCHAR(12) NOT NULL, feedback_date DATE, rating SMALLINT,
  feedback_text TEXT, sentiment_label VARCHAR(15),
  sentiment_validated VARCHAR(15), sentiment_mismatch BOOLEAN,
  CONSTRAINT pk_customer_feedback PRIMARY KEY (feedback_id),
  CONSTRAINT fk_feedback_customer FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
);

CREATE TABLE churn_labels (
  customer_id VARCHAR(12) NOT NULL, churned BOOLEAN NOT NULL, churn_date DATE, churn_reason VARCHAR(80),
  CONSTRAINT pk_churn_labels PRIMARY KEY (customer_id),
  CONSTRAINT fk_churn_customer FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
);

CREATE INDEX idx_subscriptions_customer ON subscriptions (customer_id);
CREATE INDEX idx_subscriptions_plan ON subscriptions (plan_id);
CREATE INDEX idx_subscriptions_status ON subscriptions (subscription_status);
CREATE INDEX idx_viewing_customer ON viewing_activity (customer_id);
CREATE INDEX idx_viewing_content ON viewing_activity (content_id);
CREATE INDEX idx_viewing_date ON viewing_activity (viewing_date);
CREATE INDEX idx_payments_customer ON payments (customer_id);
CREATE INDEX idx_payments_date ON payments (payment_date);
CREATE INDEX idx_payments_status ON payments (payment_status);
CREATE INDEX idx_tickets_customer ON support_tickets (customer_id);
CREATE INDEX idx_tickets_category ON support_tickets (issue_category);
CREATE INDEX idx_tickets_date ON support_tickets (ticket_date);
CREATE INDEX idx_feedback_customer ON customer_feedback (customer_id);
CREATE INDEX idx_feedback_sentiment ON customer_feedback (sentiment_label);
CREATE INDEX idx_customers_country ON customers (country);
CREATE INDEX idx_customers_segment ON customers (customer_segment);

-- ML outputs are published by src/database/load_ml_tables.py: customer_features, customer_scores, view v_customer_360.
-- Read-only role used by the copilot (created once by an admin):
--   CREATE ROLE ci_readonly LOGIN PASSWORD 'ci_readonly'; GRANT SELECT ON ALL TABLES IN SCHEMA public TO ci_readonly;
