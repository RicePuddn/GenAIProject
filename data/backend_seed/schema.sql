CREATE TABLE IF NOT EXISTS customers (
  id TEXT PRIMARY KEY, first_name TEXT NOT NULL, last_name TEXT NOT NULL,
  email TEXT UNIQUE NOT NULL, phone TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL,
  loyalty_tier TEXT NOT NULL CHECK (loyalty_tier IN ('standard','silver','gold','platinum')),
  address JSONB NOT NULL
);
CREATE TABLE IF NOT EXISTS products (
  id TEXT PRIMARY KEY, sku TEXT UNIQUE NOT NULL, name TEXT NOT NULL,
  category TEXT NOT NULL, unit_price NUMERIC(10,2) NOT NULL CHECK (unit_price >= 0)
);
CREATE TABLE IF NOT EXISTS orders (
  id TEXT PRIMARY KEY, customer_id TEXT NOT NULL REFERENCES customers(id),
  placed_at TIMESTAMPTZ NOT NULL, status TEXT NOT NULL, subtotal NUMERIC(10,2) NOT NULL,
  shipping_fee NUMERIC(10,2) NOT NULL, tax NUMERIC(10,2) NOT NULL,
  total NUMERIC(10,2) NOT NULL, currency CHAR(3) NOT NULL DEFAULT 'SGD',
  channel TEXT NOT NULL, shipping_address JSONB NOT NULL
);
CREATE TABLE IF NOT EXISTS order_items (
  id TEXT PRIMARY KEY, order_id TEXT NOT NULL REFERENCES orders(id),
  product_id TEXT NOT NULL REFERENCES products(id), quantity INT NOT NULL CHECK (quantity > 0),
  unit_price NUMERIC(10,2) NOT NULL, item_status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS shipments (
  id TEXT PRIMARY KEY, order_id TEXT NOT NULL REFERENCES orders(id),
  carrier TEXT NOT NULL, tracking_number TEXT UNIQUE NOT NULL, status TEXT NOT NULL,
  shipped_at TIMESTAMPTZ, estimated_delivery DATE, delivered_at TIMESTAMPTZ,
  destination JSONB NOT NULL
);
CREATE TABLE IF NOT EXISTS shipment_events (
  id TEXT PRIMARY KEY, shipment_id TEXT NOT NULL REFERENCES shipments(id),
  occurred_at TIMESTAMPTZ NOT NULL, status TEXT NOT NULL, location TEXT NOT NULL, details TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS payments (
  id TEXT PRIMARY KEY, order_id TEXT NOT NULL REFERENCES orders(id),
  provider TEXT NOT NULL, method TEXT NOT NULL, status TEXT NOT NULL,
  amount NUMERIC(10,2) NOT NULL, transaction_ref TEXT UNIQUE NOT NULL, paid_at TIMESTAMPTZ
);
CREATE TABLE IF NOT EXISTS returns (
  id TEXT PRIMARY KEY, order_id TEXT NOT NULL REFERENCES orders(id),
  status TEXT NOT NULL, reason TEXT NOT NULL, requested_at TIMESTAMPTZ NOT NULL,
  received_at TIMESTAMPTZ, resolution TEXT
);
CREATE TABLE IF NOT EXISTS refunds (
  id TEXT PRIMARY KEY, order_id TEXT NOT NULL REFERENCES orders(id),
  payment_id TEXT NOT NULL REFERENCES payments(id), status TEXT NOT NULL,
  amount NUMERIC(10,2) NOT NULL, reason TEXT NOT NULL, requested_at TIMESTAMPTZ NOT NULL,
  processed_at TIMESTAMPTZ, transaction_ref TEXT UNIQUE
);
CREATE TABLE IF NOT EXISTS support_tickets (
  id TEXT PRIMARY KEY, customer_id TEXT NOT NULL REFERENCES customers(id),
  order_id TEXT REFERENCES orders(id), created_at TIMESTAMPTZ NOT NULL,
  channel TEXT NOT NULL, category TEXT NOT NULL, priority TEXT NOT NULL,
  status TEXT NOT NULL, subject TEXT NOT NULL, customer_message TEXT NOT NULL,
  resolution TEXT, assigned_team TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_orders_customer ON orders(customer_id, placed_at DESC);
CREATE INDEX IF NOT EXISTS ix_shipments_order ON shipments(order_id);
CREATE INDEX IF NOT EXISTS ix_tickets_customer ON support_tickets(customer_id, created_at DESC);
